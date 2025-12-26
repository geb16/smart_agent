import threading


def fire_and_forget(fn, *args, **kwargs):
    thread = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    thread.start()
