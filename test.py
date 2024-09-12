"""This is a test module.
"""

def test(name: str) -> None:
    """Gives a greeting to the user.

    Args:
        name (str): Your name.
    """
    print(f'Hello, World! {name}')


if __name__ == '__main__':
    print('Hello, World!')