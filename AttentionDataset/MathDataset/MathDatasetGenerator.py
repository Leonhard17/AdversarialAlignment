from pathlib import Path
import random
from typing import List, Tuple


def generate_math_problems(
    num_problems: int = 1000,
    operations: List[str] = None,
    max_value: int = 100,
    seed: int = None
) -> Tuple[List[str], List[str]]:
    """
    Generate simple math problems and their solutions.

    Args:
        num_problems: Number of problems to generate.
        operations: List of operations to include ('+', '-', '*', '/').
        max_value: Maximum integer value for operands.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple of two lists: (problem_strings, solution_strings).
    """
    if operations is None:
        operations = ['+', '-', '*', '/']
    if seed is not None:
        random.seed(seed)

    problems, solutions = [], []
    for _ in range(num_problems):
        a = random.randint(1, max_value)
        b = random.randint(1, max_value)
        op = random.choice(operations)

        # Prepare question and compute answer
        if op == '/':  # ensure integer division
            question_val = a * b
            question = f"{question_val} {op} {b}"
            answer = str(question_val // b)
        else:
            question = f"{a} {op} {b}"
            answer = str(eval(question))

        # Append ' =' marker
        problems.append(f"{question} =")
        solutions.append(answer)

    return problems, solutions


def save_problems_and_solutions(
    problems: List[str],
    solutions: List[str],
    problems_path: Path,
    solutions_path: Path
) -> None:
    """
    Save lists of problems and solutions to text files.

    Args:
        problems: List of problem strings.
        solutions: List of solution strings.
        problems_path: Path to save the problems file.
        solutions_path: Path to save the solutions file.
    """
    # Ensure directory exists
    problems_path.parent.mkdir(parents=True, exist_ok=True)
    solutions_path.parent.mkdir(parents=True, exist_ok=True)

    # Write problems
    with problems_path.open('w') as pf:
        pf.write("\n".join(problems))
        pf.write("\n")

    # Write solutions
    with solutions_path.open('w') as sf:
        sf.write("\n".join(solutions))
        sf.write("\n")


def main() -> None:
    """
    Example usage: generate and save 20000 problems and solutions.
    """
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    problems_file = data_dir / 'math_problems.txt'
    solutions_file = data_dir / 'math_solutions.txt'

    # Generate problems
    problems, solutions = generate_math_problems(
        num_problems=20000,
        operations=['+', '-', '*', '/'],
        max_value=100,
        seed=42
    )

    # Save to disk
    save_problems_and_solutions(
        problems=problems,
        solutions=solutions,
        problems_path=problems_file,
        solutions_path=solutions_file
    )

    print(f"Saved {len(problems)} problems to {problems_file}")
    print(f"Saved {len(solutions)} solutions to {solutions_file}")


if __name__ == '__main__':
    main()
