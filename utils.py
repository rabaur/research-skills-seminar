import random


def use_foot_gun():
    """
    If you are feeling lucky
    """
    d = random.randint(0, 1)
    return 42 / d


if __name__ == "__main__":
    use_foot_gun()

