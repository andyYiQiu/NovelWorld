"""Standard-library opener for internal regular files, not a filesystem sandbox."""
import os
import stat


def check_sqlite_files(database):
    """Reject linked/special SQLite files before its blocking file operations."""
    for path in (database, *(database.with_name(database.name + suffix)
                            for suffix in ('-journal', '-wal', '-shm'))):
        if path.is_symlink():
            raise ValueError("SQLite 数据库及其辅助文件不能是符号链接：" + str(path))
        if path.exists() and not path.is_file():
            raise ValueError("SQLite 数据库及其辅助文件必须是普通文件：" + str(path))


def regular_opener(path, flags):
    """Do not follow the final symlink or wait for a FIFO reader/writer."""
    descriptor = os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("运行文件必须是普通文件：" + str(path))
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor
