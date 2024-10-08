
# curl -X POST -H "Content-Type: application/json" -d {"text":"Hello how are you"} https://9a3e-34-143-216-221.ngrok-free.app/llm_complete




import logging
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.callbacks import CallbackManager
from llama_index.legacy.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
)
from llama_index.legacy.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.legacy.llms.base import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.legacy.llms.custom import CustomLLM
from llama_index.legacy.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.legacy.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.legacy.prompts.base import PromptTemplate
from llama_index.legacy.types import BaseOutputParser, PydanticProgramMode

DEFAULT_HUGGINGFACE_MODEL = "StabilityAI/stablelm-tuned-alpha-3b"
if TYPE_CHECKING:
    try:
        from huggingface_hub import AsyncInferenceClient, InferenceClient
        from huggingface_hub.hf_api import ModelInfo
        from huggingface_hub.inference._types import ConversationalOutput
    except ModuleNotFoundError:
        AsyncInferenceClient = Any
        InferenceClient = Any
        ConversationalOutput = dict
        ModelInfo = Any

logger = logging.getLogger(__name__)


class HuggingFaceLLM(CustomLLM):
    """HuggingFace LLM."""

    model_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The model name to use from HuggingFace. "
            "Unused if `model` is passed in directly."
        ),
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of tokens available for input.",
        gt=0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    system_prompt: str = Field(
        default="",
        description=(
            "The system prompt, containing any extra instructions or context. "
            "The model card on HuggingFace should specify if this is needed."
        ),
    )
    query_wrapper_prompt: PromptTemplate = Field(
        default=PromptTemplate("{query_str}"),
        description=(
            "The query wrapper prompt, containing the query placeholder. "
            "The model card on HuggingFace should specify if this is needed. "
            "Should contain a `{query_str}` placeholder."
        ),
    )
    tokenizer_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The name of the tokenizer to use from HuggingFace. "
            "Unused if `tokenizer` is passed in directly."
        ),
    )
    device_map: str = Field(
        default="auto", description="The device_map to use. Defaults to 'auto'."
    )
    stopping_ids: List[int] = Field(
        default_factory=list,
        description=(
            "The stopping ids to use. "
            "Generation stops when these token IDs are predicted."
        ),
    )
    tokenizer_outputs_to_remove: list = Field(
        default_factory=list,
        description=(
            "The outputs to remove from the tokenizer. "
            "Sometimes huggingface tokenizers return extra inputs that cause errors."
        ),
    )
    tokenizer_kwargs: dict = Field(
        default_factory=dict, description="The kwargs to pass to the tokenizer."
    )
    model_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during initialization.",
    )
    generate_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during generation.",
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
            LLMMetadata.__fields__["is_chat_model"].field_info.description
            + " Be sure to verify that you either pass an appropriate tokenizer "
            "that can convert prompts to properly formatted chat messages or a "
            "`messages_to_prompt` that does so."
        ),
    )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _stopping_criteria: Any = PrivateAttr()

    def __init__(
        self,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}",
        tokenizer_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device_map: Optional[str] = "auto",
        stopping_ids: Optional[List[int]] = None,
        tokenizer_kwargs: Optional[dict] = None,
        tokenizer_outputs_to_remove: Optional[list] = None,
        model_kwargs: Optional[dict] = None,
        generate_kwargs: Optional[dict] = None,
        is_chat_model: Optional[bool] = False,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: str = "",
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        # """Initialize params."""
        # try:
        #     # import torch
        #     from transformers import (
        #         AutoModelForCausalLM,
        #         AutoTokenizer,
        #         StoppingCriteria,
        #         StoppingCriteriaList,
        #     )
        # except ImportError as exc:
        #     raise ImportError(
        #         f"{type(self).__name__} requires torch and transformers packages.\n"
        #         "Please install both with `pip install transformers[torch]`."
        #     ) from exc

        # model_kwargs = model_kwargs or {}
        # # self._model = model or AutoModelForCausalLM.from_pretrained(
        # #     model_name, device_map=device_map, **model_kwargs
        # # )

        # # # check context_window
        # # config_dict = self._model.config.to_dict()
        # model_context_window = int(
        #     config_dict.get("max_position_embeddings", context_window)
        # )
        # if model_context_window and model_context_window < context_window:
        #     logger.warning(
        #         f"Supplied context_window {context_window} is greater "
        #         f"than the model's max input size {model_context_window}. "
        #         "Disable this warning by setting a lower context_window."
        #     )
        #     context_window = model_context_window

        # tokenizer_kwargs = tokenizer_kwargs or {}
        # if "max_length" not in tokenizer_kwargs:
        #     tokenizer_kwargs["max_length"] = context_window

        # self._tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        #     tokenizer_name, **tokenizer_kwargs
        # )

        # if tokenizer_name != model_name:
        #     logger.warning(
        #         f"The model `{model_name}` and tokenizer `{tokenizer_name}` "
        #         f"are different, please ensure that they are compatible."
        #     )

        # # setup stopping criteria
        # stopping_ids_list = stopping_ids or []

        # class StopOnTokens(StoppingCriteria):
        #     def __call__(
        #         self,
        #         input_ids: torch.LongTensor,
        #         scores: torch.FloatTensor,
        #         **kwargs: Any,
        #     ) -> bool:
        #         for stop_id in stopping_ids_list:
        #             if input_ids[0][-1] == stop_id:
        #                 return True
        #         return False

        # self._stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        if isinstance(query_wrapper_prompt, str):
            query_wrapper_prompt = PromptTemplate(query_wrapper_prompt)

        messages_to_prompt = messages_to_prompt or self._tokenizer_messages_to_prompt

        super().__init__(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            device_map=device_map,
            stopping_ids=stopping_ids or [],
            tokenizer_kwargs=tokenizer_kwargs or {},
            tokenizer_outputs_to_remove=tokenizer_outputs_to_remove or [],
            model_kwargs=model_kwargs or {},
            generate_kwargs=generate_kwargs or {},
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFace_LLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
            is_chat_model=self.is_chat_model,
        )

    def _tokenizer_messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Use the tokenizer to convert messages to prompt. Fallback to generic."""
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages_dict = [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ]
            tokens = self._tokenizer.apply_chat_template(messages_dict)
            return self._tokenizer.decode(tokens)

        return generic_messages_to_prompt(messages)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:



               # request to the google colab
        import requests

        # The API endpoint
        # env var
        server_url = "https://66e2-34-173-202-254.ngrok-free.app"
        url = "{server_url}/llm_complete".format(server_url=server_url)

        # Data to be sent
        data = {
                'text': prompt
                }

        # A POST request to the API
        response = requests.post(url, json=data).json()
        # print(response.json())
        completion = response['text']
        print(completion)
        tokens = response['raw']
        print(completion)



        return CompletionResponse(text=completion, raw={"model_output": tokens})

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming completion endpoint."""
        from transformers import TextIteratorStreamer

        full_prompt = prompt
        if not formatted:
            if self.query_wrapper_prompt:
                full_prompt = self.query_wrapper_prompt.format(query_str=prompt)
            if self.system_prompt:
                full_prompt = f"{self.system_prompt} {full_prompt}"

        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        inputs = inputs.to(self._model.device)

        # remove keys from the tokenizer if needed, to avoid HF errors
        for key in self.tokenizer_outputs_to_remove:
            if key in inputs:
                inputs.pop(key, None)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            decode_kwargs={"skip_special_tokens": True},
        )
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            stopping_criteria=self._stopping_criteria,
            **self.generate_kwargs,
        )

        # generate in background thread
        # NOTE/TODO: token counting doesn't work with streaming
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        # create generator based off of streamer
        def gen() -> CompletionResponseGen:
            text = ""
            for x in streamer:
                text += x
                yield CompletionResponse(text=text, delta=x)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)


def chat_messages_to_conversational_kwargs(
    messages: Sequence[ChatMessage],
) -> Dict[str, Any]:
    """Convert ChatMessages to keyword arguments for Inference API conversational."""
    if len(messages) % 2 != 1:
        raise NotImplementedError("Messages passed in must be of odd length.")
    last_message = messages[-1]
    kwargs: Dict[str, Any] = {
        "text": last_message.content,
        **last_message.additional_kwargs,
    }
    if len(messages) != 1:
        kwargs["past_user_inputs"] = []
        kwargs["generated_responses"] = []
        for user_msg, assistant_msg in zip(messages[::2], messages[1::2]):
            if (
                user_msg.role != MessageRole.USER
                or assistant_msg.role != MessageRole.ASSISTANT
            ):
                raise NotImplementedError(
                    "Didn't handle when messages aren't ordered in alternating"
                    f" pairs of {(MessageRole.USER, MessageRole.ASSISTANT)}."
                )
            kwargs["past_user_inputs"].append(user_msg.content)
            kwargs["generated_responses"].append(assistant_msg.content)
    return kwargs




llm = HuggingFaceLLM()
llm.complete('Hello can you tell me who is the united states president in 1950?')