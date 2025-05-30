#include "tasksys.h"
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>
#include <queue>


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
    // You do not need to implement this method.
    return 0;
}

void TaskSystemSerial::sync() {
    // You do not need to implement this method.
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
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    std::atomic<int> taskCounter(0);

    auto thread_func = [&]() {
        while(true) {
            int taskid = taskCounter.fetch_add(1);
            if(taskid >= num_total_tasks)
                break;
            runnable->runTask(taskid, num_total_tasks);
        }
    };

    std::vector<std::thread> threads;
    for(int i = 0; i < _num_threads; i++) {
        threads.emplace_back(thread_func);
    }
    
    for(std::thread& t : threads)
        t.join();
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
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
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //

    taskCount = total_tasks = 0;
    left_tasks = -1;
    terminated = false;
    for(int i = 0; i < num_threads; i++) {
        threadPool.emplace_back([this, i]() {worker(); });
    }
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {

    terminated = true;

    for(auto& t : threadPool) {
        if(t.joinable()) {
            t.join();
        }
    }

    threadPool.clear();
    runner = nullptr;
}

void TaskSystemParallelThreadPoolSpinning::worker() {
    while(!terminated) {
        std::unique_lock<std::mutex> lock_w(lk_worker);
        int taskid = total_tasks - left_tasks;

        if(taskid >= total_tasks) {
            lock_w.unlock();
            std::this_thread::yield();
            continue;
        }
        
        left_tasks--;
        lock_w.unlock();

        runner->runTask(taskid, total_tasks);

        // printf("runnignf taskid: %d\n", taskid);
        {
            std::lock_guard<std::mutex> lock_f(lk_finish);
            taskCount++;
        }
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    
    runner = runnable;
    taskCount = 0;
    total_tasks = num_total_tasks;
    left_tasks = num_total_tasks;

    while(taskCount < num_total_tasks){
        std::this_thread::yield();
    }
    
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
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
    
    taskCount = 0;
    total_tasks = left_tasks = -1;
    terminated = false;
    for(int i = 0; i < num_threads; i++) {
        threadPool.emplace_back([this](){ worker(); });
    }
}

void TaskSystemParallelThreadPoolSleeping::worker() {
    while(true) {
        std::unique_lock<std::mutex> lock_w(lk_worker);
        cv_worker.wait(lock_w, [this](){ return left_tasks > 0 || terminated;});

        if(left_tasks == 0 && terminated){
            break;
        }

        int taskid = total_tasks - left_tasks;
        left_tasks--;
        lock_w.unlock();

        runner->runTask(taskid, total_tasks);

        {
            std::lock_guard<std::mutex> lock_f(lk_finish);
            taskCount++;
            if(taskCount == total_tasks) {
                cv_run.notify_one();
            }
        }
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

    for(auto& t : threadPool) {
        if(t.joinable()) {
            t.join();
        }
    }

    threadPool.clear();
    runner = nullptr;
    
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //

    runner = runnable;
    total_tasks = left_tasks = num_total_tasks;
    taskCount = 0;

    cv_worker.notify_all();

    {
        std::unique_lock<std::mutex> lock_f(lk_finish);
        cv_run.wait(lock_f, [this](){ return taskCount == total_tasks; });
    }
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
