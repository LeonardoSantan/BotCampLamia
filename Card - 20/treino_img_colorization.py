optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    predicted_color = model(gray_tensor)
    real_color = torch.randn_like(predicted_color)  # Simulated real color data
    loss = criterion(predicted_color, real_color)
    loss.backward()
    optimizer.step()
