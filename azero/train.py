
import torch

from azero.net import NetStorage
from azero.azero import AZeroConfig, ReplayBuffer


def train(config: AZeroConfig):
    buffer = ReplayBuffer(config)
    nets = NetStorage(config)
    net = nets.latest_network()
    net.train()
    optimizer = torch.optim.Adam(
        net.parameters(), weight_decay=config.weight_decay)
    loss_policy = torch.nn.CrossEntropyLoss()
    loss_value = torch.nn.MSELoss()
    step = nets.step
    while True:
        sum_loss = 0
        for _ in range(config.checkpoint_interval):
            optimizer.zero_grad()
            image, value, policy = buffer.load_batch(config.batch_size)
            pred_value, pred_policy_logits = net.forward(image)
            loss = loss_value(pred_value, value.to(net.device)) + \
                loss_policy(pred_policy_logits, policy.to(net.device))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            step += 1
            if step % 100 == 0:
                print(f'step: {step} loss: {loss.item()}')
        nets.save_network(step, sum_loss / config.checkpoint_interval, net)


if __name__ == '__main__':
    train(AZeroConfig())
