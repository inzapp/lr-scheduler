# Learning Rate Scheduler

Scheduling the learning rate has become a necessity, not an option, in training the deep learning model

By adjusting the lr properly, the model can learn faster and more efficiently than otherwise

This repository provides a powerful custom learning rate scheduler with the latest techniques available for keras optimizer

Here is a simple code snippet for use

```python
lr = 0.001
iterations = 100000
lr_scheduler = LRScheduler(iterations=iterations, lr=lr, policy='step')
for i in range(iterations):
    ...
    lr_scheduler.update(optimizer, i)
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = loss_function(y_true, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    ...
```

## Constant LR

Use a fixed learning rate. Same as not using scheduling

<img src="https://user-images.githubusercontent.com/43339281/203723958-406c6712-e582-4fc9-b9da-2cad7a1c4eea.png" width=800px>

## Step Decay

Step decay dramatically lowers the learning rate later in the training, allowing more detailed learning to the parts that could not be learned with constant lr

```python
lr_scheduler = LRScheduler(iterations=iterations, lr=lr, policy='step')
```

<img src="https://user-images.githubusercontent.com/43339281/203724622-2ced7b31-869a-42c5-a852-633b59015ce0.png" width=800px>

I found that using warm_up with 'step' policy makes learning more stable and faster

Warm_up is a method of learning by slowly increasing lr from 0 until the set lr is reached

This method is used to train various deep learning models and works mostly well regardless of which optimizer you use

Here's how to use warm_up

```python
lr_scheduler = LRScheduler(iterations=iterations, lr=lr, policy='step', warm_up=0.1)
```

The default warm_up value is 0.1, which is used to lr warm up by 10% of the given iterations

If the model does not train stably, it may be helpful to use a larger value, such as 0.5

The lr of the 'step' policy with a warm_up value of 0.5 is scheduled as follows

<img src="https://user-images.githubusercontent.com/43339281/203725499-2891834e-408e-450d-9663-b4e51fe1373f.png" width=800px>

## Cosine annealing with warm restart

The cosine decay policy is a method of decreasing lr using the cosine function, raising it rapidly, and then decreasing it again

It is known to learn the model quickly in a short time

```python
lr_scheduler = LRScheduler(iterations=iterations, lr=lr, policy='cosine')
```

<img src="https://user-images.githubusercontent.com/43339281/203725954-bb338d3d-6c27-4978-83fd-f7e60e3671fd.png" width=800px>

At the end of each cosine cycle, the cycle length is doubled, which can be adjusted to cycle_weight

## One cycle(super convergence)

The one cycle policy, also called super convergence, is similar to the cosine function

But unlike cosine, it's a scheduling method that has only one increase and one decrease

According to a super convergence paper, learning is up to 10 times faster than using constant lr

```python
lr_scheduler = LRScheduler(iterations=iterations, lr=lr, policy='onecycle')
```

<img src="https://user-images.githubusercontent.com/43339281/203726838-a897a7fd-36f2-4b93-88f1-5d4893e8793c.png" width=800px>
