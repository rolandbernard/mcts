
import torch

from azero.net import NetStorage
from azero.azero import AZeroConfig, ReplayBuffer


def train(config: AZeroConfig):
    """
    Continually run the training of the network based on the saved self-play games.
    """
    buffer = ReplayBuffer(config)
    nets = NetStorage(config)
    net = nets.latest_network()
    net.train()  # Put the network into training mode (important for batch normalization)
    step = nets.step
    optimizer = torch.optim.SGD(net.parameters(), lr=config.lr,
                                weight_decay=config.weight_decay, momentum=config.momentum)
    optimizer.param_groups[0]['initial_lr'] = config.lr
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.lr_step, gamma=config.lr_decay, last_epoch=step - 1)
    # Use cross entropy loss for the policy
    loss_policy = torch.nn.CrossEntropyLoss()
    # Use mean squared error loss for the value
    loss_value = torch.nn.MSELoss()
    while True:
        sum_loss = 0
        for _ in range(config.checkpoint_interval):
            optimizer.zero_grad()
            image, value, policy = buffer.load_batch(config.batch_size)
            pred_value, pred_policy_logits = net.forward(image)
            # The total loss is the sum of policy and value loss
            loss = loss_value(pred_value, value.to(net.device)) + \
                loss_policy(pred_policy_logits, policy.to(net.device))
            loss.backward()
            optimizer.step()
            lr_schedule.step()
            sum_loss += loss.item()
            step += 1
            if step % 100 == 0:
                print(f'step: {step} loss: {loss.item()}')
        # Save a checkpoint, with the average loss since the last checkpoint.
        nets.save_network(step, sum_loss / config.checkpoint_interval, net)


if __name__ == '__main__':
    train(AZeroConfig())
