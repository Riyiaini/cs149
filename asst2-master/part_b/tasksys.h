#ifndef _TASKSYS_H
#define _TASKSYS_H

#define MAX_TASKS 1000000

#include "itasksys.h"
#include <thread>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <atomic>
#include <queue>

/*
 * TaskSystemSerial: This class is the student's implementation of a
 * serial task execution engine.  See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemSerial: public ITaskSystem {
    public:
        TaskSystemSerial(int num_threads);
        ~TaskSystemSerial();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelSpawn: This class is the student's implementation of a
 * parallel task execution engine that spawns threads in every run()
 * call.  See definition of ITaskSystem in itasksys.h for documentation
 * of the ITaskSystem interface.
 */
class TaskSystemParallelSpawn: public ITaskSystem {
    public:
        TaskSystemParallelSpawn(int num_threads);
        ~TaskSystemParallelSpawn();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSpinning: This class is the student's
 * implementation of a parallel task execution engine that uses a
 * thread pool. See definition of ITaskSystem in itasksys.h for
 * documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSpinning: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSpinning(int num_threads);
        ~TaskSystemParallelThreadPoolSpinning();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
};

/*
 * TaskSystemParallelThreadPoolSleeping: This class is the student's
 * optimized implementation of a parallel task execution engine that uses
 * a thread pool. See definition of ITaskSystem in
 * itasksys.h for documentation of the ITaskSystem interface.
 */
class TaskSystemParallelThreadPoolSleeping: public ITaskSystem {
    public:
        TaskSystemParallelThreadPoolSleeping(int num_threads);
        ~TaskSystemParallelThreadPoolSleeping();
        const char* name();
        void run(IRunnable* runnable, int num_total_tasks);
        TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                const std::vector<TaskID>& deps);
        void sync();
    private:
        std::vector<std::thread> threadPool;
        std::mutex lk_worker;
        std::mutex lk_taskque;
        std::mutex lk_waitstk;
        std::mutex lk_done;
        std::mutex lk_run;
        std::condition_variable cv_worker;
        std::condition_variable cv_finish;
        std::atomic<bool> terminated;
        std::atomic<TaskID> next_task_id;
        struct task_t {
            int task_id;
            int taskCount;
            int num_total_tasks;
            IRunnable* runnable;
            std::vector<TaskID> deps;
            void run() {
                runnable->runTask(taskCount, num_total_tasks);
            }
            bool isnull() const {
                return task_id == -1;
            }
        };
        std::queue<task_t> taskQueue;
        std::vector<task_t> waitStack;
        task_t getTask();
        bool isReady(task_t& task);
        int done[MAX_TASKS];
        bool running[MAX_TASKS];
        void worker(int id);
};

#endif
