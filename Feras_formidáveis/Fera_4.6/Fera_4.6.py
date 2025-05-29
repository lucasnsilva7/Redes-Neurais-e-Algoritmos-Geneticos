import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets  
import torchvision.transforms as transforms
import time


df_treino = datasets.MNIST(root= 'dataset/' , train= True , transform=transforms.ToTensor(), download= False) 
df_teste = datasets.MNIST(root= 'dataset/' , train= False , transform=transforms.ToTensor(), download= False) 


class CNN(nn.Module):
    def __init__(self): 
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)    
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.fc1 = nn.Linear(14*14*8, 16) 
        self.fc2 = nn.Linear(16, 10) 
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 14*14*8)
        x = F.relu(self.fc1(x))
        return x
    
    
BATCH_SIZE = 32 # Número de amostras de imagens processadas ao mesmo tempo a cada iteração


loader_treino = DataLoader(dataset=df_treino, batch_size=BATCH_SIZE, shuffle= True) 
loader_teste = DataLoader(dataset=df_teste, batch_size=BATCH_SIZE, shuffle= True)


TAXA_APRENDIZADO = 0.001
NUM_EPOCAS = 10 # A cada época a rede neural terá passado por todas as imagens do dataset

modelo = CNN()


perda = nn.CrossEntropyLoss() # Mede a dissimilaridade entra a distribuição de rótulos verdadeira e as previsões do modelo
otimizador = optim.Adam(modelo.parameters(), lr= TAXA_APRENDIZADO)


inicio = time.time()
for epoca in range(NUM_EPOCAS):
    for images, labels in loader_treino:
        outputs = modelo(images)
        loss = perda(outputs, labels)
        otimizador.zero_grad()
        loss.backward()
        otimizador.step()
        
    print(f"Época {epoca + 1} concluída")

fim = time.time()
print()
print(f"Treinamento finalizado após {(fim - inicio) / 60} minutos")


num_acertos = 0
num_total = 0

for imagens, rotulos in loader_teste:
    imagens = imagens
    rotulos = rotulos

    saida = modelo(imagens)                      # Passa as imagens pela CNN
    previsoes = torch.argmax(saida, dim=1)       # Pega a classe com maior probabilidade

    acertos = (previsoes == rotulos).sum().item()  # Conta quantos acertaram
    num_acertos += acertos
    num_total += rotulos.size(0)                  # Soma quantas imagens tinham no batch

# Depois do loop:
acuracia = num_acertos / num_total
print(f'Acurácia: {acuracia * 100:.2f}%')
