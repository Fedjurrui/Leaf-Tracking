import torch

def trainEpochs(model,
                data_loader,
                optimizer,
                lr_scheduler=None,
                num_epopchs=10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the model using our helper function

    model.to(device)

    # construct an optimizer

    # let's train it for 10 epochs
    for epoch in range(num_epopchs):
        # train for one epoch, printing every 10 iterations
        print(f"Starting epoch {epoch}...")
        acumm_loss = 0
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            acumm_loss += losses.cpu().item()
            if i % 10 == 0:
                print(f"({i}/{len(data_loader)}) Loss: {acumm_loss/10}")
                acumm_loss = 0

        lr_scheduler.step()