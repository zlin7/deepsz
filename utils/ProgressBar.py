import time, sys

class ProgressBar:
    def __init__(self, iterable, taskname=None, barLength=40, stride = 500):
        self.l = iterable
        try:
            self.n = len(self.l)
        except TypeError:
            self.l = list(self.l)
            self.n = len(self.l)
        self.cur = 0
        self.starttime = time.time()
        self.barLength = barLength
        self.taskname = taskname
        self.last_print_time = time.time()
        self.stride = stride

    def __iter__(self):
        return self
    def _update_progress(self):
        status = "Done...\r\n" if self.cur == self.n else "\r"
        progress = float(self.cur) / self.n
        curr_time = time.time()

        block = int(round(self.barLength * progress))
        text = "{}Percent: [{}] {:.2%} Used Time:{:.2f} seconds {}".format("" if self.taskname is None else "Working on {}. ".format(self.taskname),
                                                                      "#" * block + "-"*(self.barLength - block),
                                                                      progress, curr_time - self.starttime, status)
        sys.stdout.write(text)
        sys.stdout.flush()

    def __next__(self):
        if self.cur % self.stride == 0:
            self._update_progress()
        if self.cur >= self.n:
            raise StopIteration
        else:
            self.cur += 1
            return self.l[self.cur - 1]

def test():
    for i in ProgressBar(range(100000)):
        i = i +1