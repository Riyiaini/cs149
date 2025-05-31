#include "tasksys.h"
#include <thread>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <fstream>

IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) : _num_threads(num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    next_task_id = 0;
    terminated = false;

    for (int i = 0; i < MAX_TASKS; i++) {
        done[i] = 0;
        running[i] = false;
    }

    for (int i = 0; i < num_threads; i++) {
        threadPool.emplace_back([this]() { worker(); });
    }
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    terminated = true;
    cv_worker.notify_all();
    for (auto& t : threadPool) {
        if (t.joinable()) {
            t.join();
        }
    }
    threadPool.clear();
}

void TaskSystemParallelThreadPoolSleeping::worker() {
    while (true) {
        task_t task;
        std::unique_lock<std::mutex> lock_w(lk_worker);
        cv_worker.wait(lock_w, [this, &task]() {
            task = getTask(); 
            return !task.isnull() || terminated; 
        });

        if (terminated && task.isnull()) {
            break;
        }

        lock_w.unlock();
        task.run();
            
        {
            std::lock_guard<std::mutex> lock_done(lk_done);
            ++done[task.task_id];
            if (done[task.task_id] == task.num_total_tasks) {
                std::lock_guard<std::mutex> lock_run(lk_run);
                running[task.task_id] = false;
                cv_finish.notify_all();
            }
        }

    }
}

bool TaskSystemParallelThreadPoolSleeping::isReady(task_t& task) {
    std::lock_guard<std::mutex> lock_run(lk_run);
    for (const TaskID& dep : task.deps) {
        if (running[dep]) {
            return false;
        }
    }
    return true;
}

TaskSystemParallelThreadPoolSleeping::task_t TaskSystemParallelThreadPoolSleeping::getTask() {
    std::unique_lock<std::mutex> lock_t(lk_taskque);
    if (!taskQueue.empty()) {
        auto& task = taskQueue.front();
        task_t returnTask = task;
        if(++task.taskCount == task.num_total_tasks) {
            taskQueue.pop();
        }
        return returnTask;
    }
    std::unique_lock<std::mutex> lock_w(lk_waitstk);
    if (!waitStack.empty()) {
        task_t returnTask = {-1, -1, -1, nullptr, {}};
        for (auto it = waitStack.begin(); it != waitStack.end(); ) {
            if (isReady(*it)) {
                task_t readyTask = *it;
                it = waitStack.erase(it); 
                if (returnTask.isnull()) {
                    returnTask = readyTask;
                }
                if (++readyTask.taskCount < readyTask.num_total_tasks) {
                    taskQueue.push(readyTask);
                }
                cv_worker.notify_all();
                return returnTask;
            } else {
                ++it;
            }
        }
    }
    return {-1, -1, -1, nullptr, {}};
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    std::vector<TaskID> noDeps;

    runAsyncWithDeps(runnable, num_total_tasks, noDeps);

    sync();
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    TaskID task_id = next_task_id.fetch_add(1);


    struct task_t task;
    bool is_ready = true;
    task.task_id = task_id;
    task.taskCount = 0;
    task.num_total_tasks = num_total_tasks > 0 ? num_total_tasks : 1;
    task.runnable = runnable;
    task.deps = deps;
    
    {
        std::lock_guard<std::mutex> lock_run(lk_run);
        running[task_id] = true;
        for (const TaskID& dep : deps) {
            if (running[dep]) {
                is_ready = false;
            }
        }
    }

    if (is_ready) {
        std::lock_guard<std::mutex> lock_task(lk_taskque);
        taskQueue.push(task);
    } else {
        std::lock_guard<std::mutex> lock_wait(lk_waitstk);
        waitStack.push_back(task);
    }
    
    cv_worker.notify_all();

    return task_id;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    std::unique_lock<std::mutex> lock_f(lk_run);
    cv_finish.wait(lock_f, [this]() {
        for (int i = 0; i < next_task_id; i++) {
            if (running[i]) {
                return false;
            }
        }
        return true;
    });

    return;
}
