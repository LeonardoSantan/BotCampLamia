lockdown = True
grana = 30

status = 'Em Casa' if lockdown and grana <= 100 else 'Uhull!'



print(f' O status é: {status}')