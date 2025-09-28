import random
import string
from datetime import datetime
from itertools import product

from inspect_ai import Task, eval_set, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Target,
    mean,
    metric,
    scorer,
)
from inspect_ai.solver import Generate, TaskState, solver

PROMPT = r"""You are an AI assistant.
I will provide you with a dictionary and then give you keys.
Your task is to keep a running total (starting from 0) by adding the values
associated with the keys I provide. Each turn, respond with the current running sum.

Example dictionary:
'apple': 5, 'banana': 0, 'cherry': 7, 'grape': -4, 'kiwi': 2, 'mango': -1

Example 1:
User: apple, banana
Assistant: 5
User: cherry, grape
Assistant: 8
User: kiwi, mango
Assistant: 9

Example 2:
User: apple, banana, cherry
Assistant: 12
User: grape, kiwi, mango
Assistant: 9

Example 3:
User: apple, banana, cherry, grape, kiwi, mango
Assistant: 9

Now, here is the actual task.
Dictionary to maintain:
{dictionary}

Ready to start!
IMPORTANT: DO NOT OUTPUT ANY OTHER TEXT OTHER THAN THE RUNNING SUM OF ALL TURNS.
"""


KEY_CHARACTERS = string.ascii_lowercase


@task
def retrive_and_compose(
    dictionary_size: int,
    key_length: int,
    low: int,
    high: int,
    n_turns: int,
    keys_per_turn: int,
    seed: int = 42,
) -> Task:
    return Task(
        dataset=[
            Sample(
                input=PROMPT,
                metadata={
                    "turns": {},
                    "cumsum": 0,
                    "prev_cumsum_guess": 0,
                },
            )
        ],
        solver=[
            create_dictionary(dictionary_size, key_length, low, high, seed),
            update_prompt_with_dictionary(),
            create_plan(n_turns, keys_per_turn, seed),
            play_turns(),
        ],
        scorer=[
            max_turn_absolutely_correct(),
            pct_turns_absolutely_correct(),
            pct_turns_relatively_correct(),
            turn_absolutely_correct(),
            turn_relatively_correct(),
        ],
    )


@solver
def create_dictionary(
    n: int = 100, k: int = 5, low: int = -99, high: int = 99, seed: int = 42
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: D417
        """
        Creates a dictionary of size with random string keys and random integer values.

        Args:
            n (int): The size of the dictionary to create (default: 100)
            k (int): The number of characters in the random string keys (default: 5)
            low (int): The minimum value for random integer values (default: -99)
            high (int): The maximum value for random integer values (default: 99)
            seed (int): Random seed for reproducible dictionary generation (default: 42)

        Returns:
            TaskState: Updated state with dictionary in metadata
        """
        max_possible_keys = len(KEY_CHARACTERS) ** k
        if n > max_possible_keys:
            raise ValueError(f"Cannot create {n} unique strings from {k} characters")

        if low > high:
            raise ValueError(f"low ({low}) must be <= high ({high})")

        random.seed(seed)  # Ensures same dictionary across all models
        dictionary = {}
        while len(dictionary) < n:
            key = "".join(random.choices(KEY_CHARACTERS, k=k))
            value = random.randint(low, high)
            dictionary[key] = value

        state.metadata["dictionary"] = dictionary
        return state

    return solve


@solver
def update_prompt_with_dictionary():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        dictionary = state.metadata["dictionary"]
        state.messages[0].content = PROMPT.format(dictionary=dictionary)
        return state

    return solve


@solver
def create_plan(n: int, k: int, seed: int = 42):
    async def solve(state: TaskState, generate: Generate) -> TaskState:  # noqa: D417
        """
        Updates state with a plan consisting of n turns, where each turn contains a random selection of k keys.

        Args:
            n (int): The number of turns in the plan
            k (int): The number of keys to select in each turn, which can be thought of as
                the number of steps per turn.
            seed (int): Random seed for reproducible plan generation (default: 42)

        Returns:
            TaskState: Updated state with plan in metadata
        """  # noqa: E501
        random.seed(seed)  # ensure same plan across all models
        dictionary = state.metadata["dictionary"]
        state.metadata["plan"] = [
            random.choices(list(dictionary.keys()), k=k) for _ in range(n)
        ]
        return state

    return solve


@solver
def play_turns():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        plan = state.metadata.get("plan")
        dictionary = state.metadata.get("dictionary")

        def calculate_error(correct_value, guess):
            return abs(correct_value - guess) if guess != float("inf") else float("inf")

        for turn, keys in enumerate(plan, start=1):
            # Send keys to model and get response
            state.messages.append(ChatMessageUser(content=f"{', '.join(keys)}"))
            await generate(state)

            # Calculate turn data
            keys_dict = {key: dictionary[key] for key in keys}
            keys_sum = sum(keys_dict.values())

            # Get previous state
            prev_cumsum = state.metadata["cumsum"]
            prev_cumsum_guess = state.metadata["prev_cumsum_guess"]

            # Update cumulative sum and calculate targets
            state.metadata["cumsum"] += keys_sum
            absolute_cumsum = state.metadata["cumsum"]
            relative_cumsum = prev_cumsum_guess + keys_sum

            # Parse model's guess
            try:
                cumsum_guess = int(state.output.completion)
            except ValueError:
                cumsum_guess = float("inf")

            # Store comprehensive turn data
            state.metadata["turns"][turn] = {
                "keys": keys_dict,
                "score": {
                    "keys_sum": keys_sum,
                    "prev_cumsum": prev_cumsum,
                    "prev_cumsum_guess": prev_cumsum_guess,
                    "absolute_cumsum": absolute_cumsum,
                    "relative_cumsum": relative_cumsum,
                    "cumsum_guess": cumsum_guess,
                    "is_absolutely_correct": absolute_cumsum == cumsum_guess,
                    "is_relatively_correct": relative_cumsum == cumsum_guess,
                    "absolute_cumsum_error": calculate_error(
                        absolute_cumsum, cumsum_guess
                    ),
                    "relative_cumsum_error": calculate_error(
                        relative_cumsum, cumsum_guess
                    ),
                },
            }

            # Update for next turn
            state.metadata["prev_cumsum_guess"] = cumsum_guess

        return state

    return solve


@metric
def max_metric() -> Metric:
    def metric(scores: list[SampleScore]) -> int:
        return max([score.score.as_int() for score in scores])

    return metric


@scorer(metrics=[max_metric()])
def max_turn_absolutely_correct():
    async def score(state: TaskState, target: Target) -> Score:
        turns = state.metadata.get("turns", {})
        correct_turns = [
            turn
            for turn, data in turns.items()
            if data["score"]["is_absolutely_correct"]
        ]
        return Score(value=max(correct_turns, default=0))

    return score


@scorer(metrics=[mean()])
def pct_turns_absolutely_correct():
    async def score(state: TaskState, target: Target) -> Score:
        turns = state.metadata.get("turns", {})
        if not turns:
            return Score(value=0.0)
        return Score(
            value=sum(data["score"]["is_absolutely_correct"] for data in turns.values())
            / len(turns)
        )

    return score


@scorer(metrics=[mean()])
def pct_turns_relatively_correct():
    async def score(state: TaskState, target: Target) -> Score:
        turns = state.metadata.get("turns", {})
        if not turns:
            return Score(value=0.0)
        return Score(
            value=sum(data["score"]["is_relatively_correct"] for data in turns.values())
            / len(turns)
        )

    return score


@scorer(metrics=[])
def turn_absolutely_correct():
    async def score(state: TaskState, target: Target) -> Score:
        turns = state.metadata.get("turns", {})
        return Score(
            value={
                str(turn): data["score"]["is_absolutely_correct"]
                for turn, data in turns.items()
            }
        )

    return score


@scorer(metrics=[])
def turn_relatively_correct():
    async def score(state: TaskState, target: Target) -> Score:
        turns = state.metadata.get("turns", {})
        return Score(
            value={
                str(turn): data["score"]["is_relatively_correct"]
                for turn, data in turns.items()
            }
        )

    return score


if __name__ == "__main__":
    params = {
        "dictionary_size": [
            100,
        ],  # [5, 10, 50, 100, 250, 500, 1_000, 10_000],
        "key_length": [5],  # [1, 2, 3, 5, 10, 25, 50, 100],
        "int_range": [
            (-99, 99),
        ],  # [(-10, 10), (-99, 99), (-999, 999), (-9_999, 9_999)],
        "n_turns": [1_000],
        "keys_per_turn": [
            1,
        ],  # [1, 2, 3, 5, 10, 25, 50, 100, 500, 1_000, 10_000],
        "seed": [42],
    }
    grid = list(product(*(params[name] for name in params)))

    eval_set(
        [
            retrive_and_compose(
                dictionary_size,
                key_length,
                *int_range,
                n_turns,
                keys_per_turn,
                seed,
            )
            for dictionary_size, key_length, int_range, n_turns, keys_per_turn, seed in grid  # noqa: E501
            # Filter for valid combinations
            if dictionary_size <= len(KEY_CHARACTERS) ** key_length
            and int_range[0] <= int_range[1]
        ],
        model=[
            # "anthropic/claude-3-haiku-20240307",
            "anthropic/claude-3-5-haiku-20241022",
            "anthropic/claude-3-7-sonnet-20250219",
            "anthropic/claude-sonnet-4-20250514",
            # "anthropic/claude-opus-4-1-20250805",
            # "openai/gpt-3.5-turbo-0125",
            # # "openai/gpt-4-turbo-2024-04-09",
            "openai/gpt-4o-2024-08-06",
            "openai/gpt-4.1-nano-2025-04-14",
            # # "openai/gpt-4.1-2025-04-14",
            # "openai/o1-2024-12-17",
            # "openai/o3-2025-04-16",
            # "openai/gpt-5-nano-2025-08-07",
            # "openai/gpt-5-2025-08-07",
        ],
        log_dir=f"logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}",
        epochs=3,
        max_tasks=30,
        max_connections=30,
        max_tokens=16,  # OAI minimum
        reasoning_tokens=None,  # disable thinking for Anthropic models
        reasoning_effort="minimal",
        reasoning_history="none",
        reasoning_summary=None,
    )
