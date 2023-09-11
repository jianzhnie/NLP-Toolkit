from typing import Tuple


# Function to calculate elapsed time
def epoch_time(start_time: int, end_time: int) -> Tuple[int, int]:
    """
    Calculate elapsed time in minutes and seconds.

    Args:
        start_time (int): Start time in seconds.
        end_time (int): End time in seconds.

    Returns:
        Tuple[int, int]: Elapsed minutes and seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
