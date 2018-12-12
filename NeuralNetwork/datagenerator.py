import numpy as np

def ReadInput(filename):
  FirstLine = True
  Titles = []
  Data = []
  with open(filename) as f:
    for lines in f:
      line = lines.strip().split(",")
      if FirstLine:
        FirstLine = False
        Titiles = line
        continue
      Data.append(line)

  Data = np.array(Data)
  print Data
  return Data

if __name__ == "__main__":
    Data = ReadInput("./train.csv")
    np.random.shuffle(Data)

    for i in range(Data.shape[0]):
        if i <= Data.shape[0]*0.33:
            with open("./data/test.csv", "a") as f:
                data = ",".join(list(Data[i]))
                f.write(data + "\n")
        else:
            with open("./data/train.csv", "a") as f:
                data = ",".join(list(Data[i]))
                f.write(data + "\n")
        
