import os


def is_directory_empty(directory_path: str) -> bool:
    """
    Check if a directory is empty.

    Args:
        directory_path (str): The path to the directory to be checked.

    Returns:
        bool: True if the directory is empty, False otherwise.
    """
    # 使用os.listdir()获取目录中的所有文件和子目录的列表
    content = os.listdir(directory_path)

    # 如果列表为空，表示目录为空
    if not content:
        return True
    else:
        return False


def get_outdir(path: str, *paths: str, inc: bool = False) -> str:
    """
    Create an output directory if it is empty and return its path. If inc is True,
    increment the directory name until a unique name is found.

    Args:
        path (str): The base directory where the output directory will be created.
        *paths (str): Additional path components to append to the base directory.
        inc (bool, optional): If True, increment the directory name with a number
            until a unique name is found. Defaults to False.

    Returns:
        str: The path of the created (or existing) output directory.

    Raises:
        AssertionError: If a unique directory name couldn't be found after 100 attempts.
    """
    # Join the base path and additional components to create the output directory path.
    outdir = os.path.join(path, *paths)

    # Check if the directory already exists.
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        # If inc is True, increment the directory name until a unique name is found.
        count = 1
        while count < 100:
            outdir_inc = f'{outdir}-{count}'
            # Check if the incremented directory exists, if not create it.
            if not os.path.exists(outdir_inc):
                os.makedirs(outdir_inc)
                outdir = outdir_inc
                break
            # If the incremented directory exists, check if it is empty, if so use it.
            elif is_directory_empty(outdir_inc):
                outdir = outdir_inc
                break
            count += 1
            # If the incremented directory exists and is not empty, increment the name again.
        else:
            # Raise an AssertionError if a unique name couldn't be found.
            raise AssertionError(
                'Unable to find a unique directory name after 100 attempts.')

    return outdir


if __name__ == '__main__':
    # Example usage:
    path = '/home/robin/work_dir/llm/nlp-toolkit/work_dirs'
    outdir = get_outdir(path, 'output_folder', inc=True)
    print(outdir)
