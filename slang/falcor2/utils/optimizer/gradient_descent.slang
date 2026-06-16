// SPDX-License-Identifier: Apache-2.0
implementing utils;

public struct GradientDescentOptimizer<T : __BuiltinFloatingPointType> : IOptimizer<T>
{
    // Dummy internal state
    public struct GDState<S : __BuiltinFloatingPointType> : IOptimizerState<S>
    {
        int dummy;

        public __init(S param) { dummy = 0; }
    }
    public typealias State = GDState<T>;

    // Batch of adam optimizer parameters, gradients and states
    public struct GDBatch<S : __BuiltinFloatingPointType>
    {
        RWStructuredBuffer<S> params;
        RWStructuredBuffer<S> grads;
    }
    public typealias Batch = GDBatch<T>;

    // GD parameters
    public T learning_rate;

    public void step(inout State state, inout T param, inout T grad)
    {
        param -= learning_rate * grad;
        grad = T(0);
    }

    public void batch_step<let N : int>(Batch[N] batches, int2 batch_index)
    {
        GDBatch<T> batch = batches[batch_index.x];
        GDState<T> state = GDState<T>(0); // dummy state, not used

        int i = batch_index.y;

        T param = batch.params[i];
        T grad = batch.grads[i];

        step(state, param, grad);

        batch.params[i] = param;
        batch.grads[i] = grad;
    }

    public void batch_zero_grads<let N : int>(Batch[N] batches, int2 batch_index)
    {
        GDBatch<T> batch = batches[batch_index.x];
        int i = batch_index.y;
        batch.grads[i] = T(0);
    }
}
