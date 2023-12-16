#ifndef SAFE_QUEUE_HPP_
#define SAFE_QUEUE_HPP_

#include <queue>
#include <mutex>

namespace mujoco_parallel_sim {

template <typename T>
class SafeQueue
{
public:
    SafeQueue() = default;

    SafeQueue &operator=(const SafeQueue& other)
    {
        std::lock_guard<std::mutex> lock(m_);
        q_ = other.q_;
        return *this;
    }

    int Size()
    {
        std::lock_guard<std::mutex> lock(m_);
        return q_.size();
    }

    bool Empty()
    {
        std::lock_guard<std::mutex> lock(m_);
        return q_.empty();
    }

    void Push(const T &val)
    {
        std::lock_guard<std::mutex> lock(m_);
        q_.push(val);
    }

    void Push(T &&val)
    {
        std::lock_guard<std::mutex> lock(m_);
        q_.push(val);
    }

    bool TryPop(T &val)
    {
        std::lock_guard<std::mutex> lock(m_);
        if (q_.empty())
            return false;
        else {
            val = q_.front();
            q_.pop();
            return true;
        }
    }

private:
    std::queue<T> q_;
    std::mutex m_;
};

} // namespace mujoco_parallel_sim

#endif // SAFE_QUEUE_HPP_
