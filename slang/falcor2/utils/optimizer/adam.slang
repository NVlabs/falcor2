// SPDX-License-Identifier: Apache-2.0
implementing utils;

public struct AdamOptimizer<T : __BuiltinFloatingPointType> : IOptimizer<T>
{
    // Internal state storing per-parameter moments
    public struct AdamState<S : __BuiltinFloatingPointType> : IOptimizerState<S>
    {
        S mean;
        S variance;
        int iteration;

        public __init(S param)
        {
            mean = S(0);
            // variance = S(-1);
            variance = S(0);
            iteration = 0;
        }
    }
    public typealias State = AdamState<T>;

    // Batch of adam optimizer parameters, gradients and states
    public struct AdamBatch<S : __BuiltinFloatingPointType>
    {
        RWStructuredBuffer<S> params;
        RWStructuredBuffer<S> grads;
        RWStructuredBuffer<AdamState<S>> states;
    }
    public typealias Batch = AdamBatch<T>;

    // Adam parameters
    public T beta1;
    public T beta2;
    public T epsilon;
    public T learning_rate;

    public void step(inout State state, inout T param, inout T grad)
    {
        // Note: The standard Adam implementation contains bias in the mean and variance
        // This is because they are a moving average and are initialized to zero,
        // leading to a bias towards zero that diminishes over time.
        // The original Adam paper suggests to cancel out the bias by dividing out
        // the weight of the zero term, computed with a power of the betas (!) and the
        // iteration count.
        // This implements a much simpler solution to cancelling the bias by initializing
        // the moving average from the gradients in the first iteration, instead of zero.
        // This avoids the bias entirely.
        // We avoid an iteration count or separate boolean flag by initializing the variance
        // part of the moments to a negative number; in normal operation, the variance can
        // only be positive, and we know we're starting from fresh initialization if the
        // variance is negative.
        // bool is_first_iteration = state.variance < T(0);
        // T blend_mean = is_first_iteration ? T(0) : beta1;
        // T blend_variance = is_first_iteration ? T(0) : beta2;
        // state.mean = lerp(grad, state.mean, blend_mean);
        // state.variance = lerp(grad * grad, state.variance, blend_variance);
        // param -= learning_rate * state.mean / (sqrt(state.variance) + epsilon);
        // grad = T(0);
        state.iteration++;
        state.mean = beta1 * state.mean + (T(1) - beta1) * grad;
        state.variance = beta2 * state.variance + (T(1) - beta2) * grad * grad;
        T mean_hat = state.mean / (T(1) - pow(beta1, T(state.iteration)));
        T variance_hat = state.variance / (T(1) - pow(beta2, T(state.iteration)));
        param -= learning_rate * mean_hat / (sqrt(max(T(0), variance_hat)) + epsilon);
        grad = T(0);
    }

    public void batch_step<let N : int>(Batch[N] batches, int2 batch_index)
    {
        AdamBatch<T> batch = batches[batch_index.x];

        int i = batch_index.y;

        T param = batch.params[i];
        T grad = batch.grads[i];
        AdamState<T> state = batch.states[i];

        step(state, param, grad);

        batch.params[i] = param;
        batch.grads[i] = grad;
        batch.states[i] = state;
    }

    public void batch_zero_grads<let N : int>(Batch[N] batches, int2 batch_index)
    {
        AdamBatch<T> batch = batches[batch_index.x];
        int i = batch_index.y;
        batch.grads[i] = T(0);
    }
}
