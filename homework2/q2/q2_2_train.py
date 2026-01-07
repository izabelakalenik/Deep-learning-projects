import torch
import torch.optim as optim
from config import RNAConfig
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial

from utils import masked_mse_loss, masked_spearman_correlation, load_rnacompete_data, plot, configure_seed
import q2_2_mhccn as mhccn_module
import q2_1_cnn as cnn_module


def comparative_plot(data_groups, out_name="comparison_plot.png"):
    plt.clf()
    for group in data_groups:
        epochs = range(1, len(group['train']) + 1)
        
        # train (solid)
        plt.plot(epochs, group['train'], label=f"{group['label']} Train", 
                 color=group['color'], lw=2)
        
        # validation (dashed)
        plt.plot(epochs, group['val'], label=f"{group['label']} Val", 
                 color=group['color'], linestyle='--', alpha=0.7)

    plt.xlabel('Epoch')
    plt.legend()
    
    #save file
    plt.savefig(out_name, bbox_inches='tight')


def train_network(model_class, model_name, protein='RBFOX1', epochs=20, batch_size=64, lr=0.001, kernel_size=7):
    print(f"\n--- Training {model_name} on {protein} ---")

    configure_seed(RNAConfig.SEED)
    
    train_dset = load_rnacompete_data(protein, split='train')
    val_dset   = load_rnacompete_data(protein, split='val')
    test_dset  = load_rnacompete_data(protein, split='test')

    gen = torch.Generator()
    gen.manual_seed(RNAConfig.SEED)
    
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, generator = gen)
    val_loader   = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=False, generator = gen)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False, generator = gen)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        batch_loss_accum = 0
        
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            
   
            loss = masked_mse_loss(preds, y, mask)
            
            loss.backward()
            optimizer.step()
            batch_loss_accum += loss.item()
        
        avg_train_loss = batch_loss_accum / len(train_loader)
        train_losses.append(avg_train_loss)


        model.eval()
        all_preds, all_targs, all_masks = [], [], []
        
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                preds = model(x)
                
                all_preds.append(preds)
                all_targs.append(y)
                all_masks.append(mask)
                
        vp = torch.cat(all_preds)
        vt = torch.cat(all_targs)
        vm = torch.cat(all_masks)
        
        val_loss = masked_mse_loss(vp, vt, vm).item()
        val_corr = masked_spearman_correlation(vp, vt, vm).item()
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Corr: {val_corr:.4f}")

    model.eval()
    test_preds, test_targs, test_masks = [], [], []
    with torch.no_grad():
        for x, y, mask in test_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            test_preds.append(model(x))
            test_targs.append(y)
            test_masks.append(mask)
            
    tp = torch.cat(test_preds)
    tt = torch.cat(test_targs)
    tm = torch.cat(test_masks)
    
    test_corr = masked_spearman_correlation(tp, tt, tm).item()
    print(f"--- Final Test Spearman Correlation: {test_corr:.4f} ---")
    
    plot(range(1, epochs+1), 
         {'Train Loss': train_losses, 'Val Loss': val_losses}, 
         filename=f"{model_name}_loss_plot.png")
    
    return test_corr, train_losses, val_losses


if __name__ == "__main__":
    LR = 0.001
    EPOCHS = 20

    # capture comparation
    _, cnn_tl, cnn_vl  = train_network(
    model_class=cnn_module.RNA_CNN, 
    model_name="rna_cnn",
    epochs=EPOCHS,
    lr=LR
    )

    # grid search for head n
    head_options = [2, 4, 8]
    results = []
    best_corr = 0
    best_vl = 0
    best_tl = 0
    best_n = 0

    for heads in head_options:
        print(f"\n Testing: {heads} heads")
        
        # pass the argument in a way enabling a call without arguments
        bound_model = partial(mhccn_module.RNA_CNN_MultiHeadAttention, num_heads=heads)
        
        # call train function
        test_corr, tl, vl = train_network(
            model_class=bound_model, 
            model_name=f"MHA_{heads}heads",
            epochs=EPOCHS,
            lr=LR
        )

        if (test_corr > best_corr):
            best_corr = test_corr
            best_tl = tl
            best_vl = vl
            best_n = heads

        
        # record the results
        results.append({
            'num_heads': heads,
            'lr': LR,
            'test_spearman': test_corr
        })

    # save to file
    df = pd.DataFrame(results)
    df.to_csv("head_test_results.csv", index=False)
    
    print("\n--- Experiment Complete ---")
    print("best heads: ", best_n, "\n")
    print(df)


    models = [
        {"label": "S-A", "color": "C0", "train": best_tl, "val": best_vl},
        {"label": "CNN", "color": "C1" , "train": cnn_tl,  "val": cnn_vl }
    ]
    # create a plot with plain cnn and best self-atteniton model
    comparative_plot(models)


