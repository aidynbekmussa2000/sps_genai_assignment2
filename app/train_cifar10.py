import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from .model import SimpleCNN

def loaders():
    tfm_tr = transforms.Compose([transforms.Resize((64,64)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor()])
    tfm_te = transforms.Compose([transforms.Resize((64,64)),
                                 transforms.ToTensor()])
    tr = datasets.CIFAR10("./data", train=True,  download=True, transform=tfm_tr)
    te = datasets.CIFAR10("./data", train=False, download=True, transform=tfm_te)
    return DataLoader(tr, batch_size=128, shuffle=True, num_workers=2), \
           DataLoader(te, batch_size=256, shuffle=False, num_workers=2)

@torch.no_grad()
def evaluate(m, dl, device):
    m.eval(); total=0; correct=0; loss_sum=0.0; crit=nn.CrossEntropyLoss()
    for x,y in dl:
        x,y = x.to(device), y.to(device)
        out = m(x); loss = crit(out,y)
        loss_sum += loss.item()*y.size(0)
        correct += (out.argmax(1)==y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

def main(epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, te = loaders()
    m = SimpleCNN(10).to(device)
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(m.parameters(), lr=lr)
    best=0.0
    Path("artifacts").mkdir(exist_ok=True)
    for e in range(1,epochs+1):
        m.train(); run=0.0
        for x,y in tr:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(); out=m(x); loss=crit(out,y)
            loss.backward(); opt.step()
            run += loss.item()*y.size(0)
        tr_loss = run/len(tr.dataset)
        val_loss, val_acc = evaluate(m, te, device)
        print(f"epoch {e}: train={tr_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f}")
        if val_acc>best:
            best=val_acc
            torch.save(m.state_dict(),"artifacts/model.pt")
            print("✓ saved artifacts/model.pt")
    print("best acc:", best)

if __name__=="__main__":
    main()