from transformers import AutoTokenizer
import torch
from vllm.logger import init_logger
import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
from vllm.inputs.data import TextPrompt, TokensPrompt

logger = init_logger("vllm")

def create_test_prompts() -> List[Tuple[TextPrompt, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    hidden_states = torch.load("./examples/hidden_states.pt")
    logger.info(f"Hidden states shape: {hidden_states.shape}")
    logger.info(f"Hidden states: {hidden_states}")

    return [
        (TextPrompt(prompt_embeds=hidden_states),
         SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1, stop=["but"]))
    ]

def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[TextPrompt, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id = 0

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                logger.info(f"Finished request output: {request_output}")
                logger.info(f"Finished generate token ids: {request_output.outputs[0].token_ids}")
                logger.info(f"Finished generated text: {request_output.outputs[0].text}")
            else:
                logger.info(f"Partial request output: {request_output}")
                logger.info(f"Partial generate token ids: {request_output.outputs[0].token_ids}")
                logger.info(f"Partial generated text: {request_output.outputs[0].text}")

def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)

def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)

if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
