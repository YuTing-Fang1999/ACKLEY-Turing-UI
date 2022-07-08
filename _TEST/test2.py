import threading
import time


def fun(cndition):
    time.sleep(1)  # 确保先运行t2
    # 获得锁
    cndition.acquire()
    print('thread1 acquires lock.')
    # 唤醒t2
    cndition.notify()
    # 进入等待状态，等待其他线程唤醒
    # cndition.wait()
    print('thread1 acquires lock again.')
    # 释放锁
    cndition.release()


def fun2(cndition):
    # 获得锁
    cndition.acquire()
    print('thread2 acquires lock.')
    # 进入等待状态，等待其他线程唤醒
    cndition.wait()
    print('thread2 acquires lock again.')
    # 唤醒t1
    # cndition.notify()
    # 释放锁
    cndition.release()


if __name__ == '__main__':
    cndition = threading.Condition()
    t1 = threading.Thread(target=fun, args=(cndition,))
    t2 = threading.Thread(target=fun2, args=(cndition,))
    t1.start()
    t2.start()