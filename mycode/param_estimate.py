import numpy as np
import matplotlib.pyplot as plt
import torch


def model(t_u, w, b):
    return w * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs


def dmodel_dw(t_u):
    return t_u


def dmodel_db():
    return 1.0


def grad_fn(t_u, t_c, t_p):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u)
    dloss_db = dloss_dtp * dmodel_db()
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


def training_loop(n_epochs, learning_rate, params, t_u, t_c, print_params=True):
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p)
        params = params - learning_rate * grad
        if epoch in {1, 2, 3, 10, 11, 99, 100, 4000, 5000}:
            print("Epoch %d, Loss %f" % (epoch, float(loss)))
            if print_params:
                print("    Params:", params)
                print("    Grad:  ", grad)
        if epoch in {4, 12, 101}:
            print("...")
        if not torch.isfinite(loss).all():
            break
    return params


def main():
    t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    t_c = torch.tensor(t_c)
    t_u = torch.tensor(t_u)

    w = torch.ones(())
    b = torch.zeros(())

    t_p = model(t_u, w, b)
    print(t_p)

    loss = loss_fn(t_p, t_c)
    print(loss)

    x = torch.ones(())
    y = torch.ones(3, 1)
    z = torch.ones(1, 3)
    a = torch.ones(2, 1, 1)
    print(f"shapes: x: {x.shape}, y: {y.shape}")
    print(f"        z: {z.shape}, a: {a.shape}")
    print("x * y:", (x * y).shape)
    print("y * z:", (y * z).shape)
    print("y * z * a:", (y * z * a).shape)

    delta = 0.1
    learning_rate = 1e-2
    loss_change_w = loss_fn(model(t_u, w + delta, b), t_c) - loss_fn(model(t_u, w - delta, b), t_c)
    loss_change_w /= 2.0 * delta
    w = w - learning_rate * loss_change_w
    loss_change_b = loss_fn(model(t_u, w, b + delta), t_c) - loss_fn(model(t_u, w, b - delta), t_c)
    loss_change_b /= 2.0 * delta
    b = b - learning_rate * loss_change_b
    print(w, b)

    training_loop(
        n_epochs=100, learning_rate=1e-4, params=torch.tensor([1.0, 0.0]), t_u=t_u, t_c=t_c
    )
    t_un = 0.1 * t_u
    training_loop(
        n_epochs=100, learning_rate=1e-2, params=torch.tensor([1.0, 0.0]), t_u=t_un, t_c=t_c
    )
    params = training_loop(
        n_epochs=5000,
        learning_rate=1e-2,
        params=torch.tensor([1.0, 0.0]),
        t_u=t_un,
        t_c=t_c,
        print_params=False,
    )

    t_p = model(t_un, *params)

    plt.xlabel("Temperature (°Fahrenheit)")
    plt.ylabel("Temperature (°Celsius)")
    plt.plot(t_u.numpy(), t_p.detach().numpy())
    plt.plot(t_u.numpy(), t_c.numpy(), "o")
    plt.savefig("temp_unknown_plot.png", format="png")


if __name__ == "__main__":
    main()
