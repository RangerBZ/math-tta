import unittest

from math_tta.prompts import tokenize_assistant_target


class BasicTokenizer:
    chat_template = None
    eos_token = "<eos>"

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        tokens = text.split()
        return {
            "input_ids": [
                self._ids.setdefault(token, len(self._ids) + 1)
                for token in tokens
            ]
        }


class PromptTests(unittest.TestCase):
    def test_masks_prompt_tokens_and_trains_on_assistant_target(self) -> None:
        tokenizer = BasicTokenizer()

        result = tokenize_assistant_target(
            tokenizer,
            "What is 2 + 2?",
            r"\boxed{4}",
            max_length=100,
        )

        self.assertIsNotNone(result)
        input_ids, labels = result
        prompt_length = len(
            tokenizer(
                "Solve the following math problem. Show your reasoning and put "
                "the final answer in \\boxed{}.\n\nWhat is 2 + 2?\n",
                add_special_tokens=True,
            )["input_ids"]
        )
        self.assertEqual(labels[:prompt_length], [-100] * prompt_length)
        self.assertEqual(labels[prompt_length:], input_ids[prompt_length:])

    def test_skips_examples_longer_than_training_context(self) -> None:
        self.assertIsNone(
            tokenize_assistant_target(
                BasicTokenizer(),
                "question",
                "a very long completion",
                max_length=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
