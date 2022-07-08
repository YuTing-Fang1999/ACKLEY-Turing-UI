from threading import *
from time import sleep


class Even:
    def __init__(self):
        self.c = Condition()

    def evenNumbers(self):
        self.c.acquire()
        for i in range(0, 101, 2):
            print("Even : ", i)
            self.num = i
            sleep(1.)
            self.c.notify()
            self.c.wait()
        self.c.release()


class Odd:
    def __init__(self, even):
        self.even = even

    def oddNumbers(self):
        self.even.c.acquire()
        for i in range(1, 100, 2):
            print("Odd : ", i)
            self.num = i
            sleep(1.)
            self.even.c.notify()
            self.even.c.wait()
        self.c.release()


tEven = Even()
tOdd = Odd(tEven)

t1 = Thread(target=tEven.evenNumbers)
t2 = Thread(target=tOdd.oddNumbers)

t1.start()
t2.start()