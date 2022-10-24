
import random
import os, pathlib, random
base_dir = pathlib.Path("aclImdb")

from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

def learn_model(trainFiles):
    
    # Directories for each author
    
    baseDir = pathlib.Path("dataset")
    trainDir = baseDir / "trainFinalDb"
    for author in ("dickens","lewis","twain"):
        path = trainDir / author
        if not os.path.exists(path):
            os.makedirs(path)

    authorName = ["dickens","lewis","twain"]
    
    # Adding data to database
    
    db = []

    # Dickens file

    dickensFile = open(trainFiles[0][0],"r", encoding="cp1252")
    dickensDb = dickensFile.read()
    dickensFile.close()
    db.append(dickensDb)

    # Lewis file

    lewisFile1 = open(trainFiles[1][0],"r", encoding="cp1252")
    lewisDb = lewisFile1.read()
    lewisFile1.close()
    lewisFile2 = open(trainFiles[1][1],"r", encoding="cp1252")
    lewisDb += lewisFile2.read()
    lewisFile2.close()
    db.append(lewisDb)

    # Twain file

    twainFile = open(trainFiles[2][0],"r", encoding="cp1252")
    twainDb = twainFile.read()
    twainFile.close()
    db.append(twainDb)

    # Creating Text Files from the database

    # Dickens Text Files

    db[0].replace("\n"," ")
    db[0].replace("?",".")
    statements = db[0].split(".")

    i = 0

    while i < 1000:
        point = random.randint(1,len(statements))
        if point == len(statements):
            point = 0
        statement = statements[point]
        j = 1
        while len(statement.split())<40:
            if (point+j) == len(statements):
                point = 0
            statement += statements[point+j]
            j = j+1    
        fileName = "%s/%s/%sFile%s.txt" %(trainDir,authorName[0],authorName[0],i)
        file = open(fileName, 'w')
        file.write(statement)
        file.close()
        i += 1

    # Lewis Text Files

    db[1].replace("\n"," ")
    db[1].replace("?",".")
    statements = db[1].split(".")

    i = 0

    while i < 1000:
        point = random.randint(1,len(statements))
        if point == len(statements):
            point = 0
        statement = statements[point]
        j = 1
        while len(statement.split())<40:
            if (point+j) == len(statements):
                point = 0
            statement += statements[point+j]
            j = j+1    
        fileName = "%s/%s/%sFile%s.txt" %(trainDir,authorName[1],authorName[1],i)
        file = open(fileName, 'w')
        file.write(statement)
        file.close()
        i += 1

    # Twain Text Files

    db[2].replace("\n"," ")
    db[2].replace("?",".")
    statements = db[2].split(".")

    i = 0

    while i < 1000:
        point = random.randint(1,len(statements))
        if point == len(statements):
            point = 0
        statement = statements[point]
        j = 1
        while len(statement.split())<40:
            if (point+j) == len(statements):
                point = 0
            statement += statements[point+j]
            j = j+1    
        fileName = "%s/%s/%sFile%s.txt" %(trainDir,authorName[2],authorName[2],i)
        file = open(fileName, 'w')
        file.write(statement)
        file.close()
        i += 1
    
    # Vectorization and Model

    bsz = 32
    train_ds = keras.utils.text_dataset_from_directory("dataset/trainFinalDb", batch_size=bsz)
    textVectorization = TextVectorization(max_tokens=10000, output_mode="multi_hot")
    text_only_train_ds = train_ds.map(lambda x, y: x)
    textVectorization.adapt(text_only_train_ds)

    model = keras.Sequential([textVectorization,
                              keras.layers.Dense(16, activation="relu"),
                              keras.layers.Dropout(0.5),
                              keras.layers.Dense(3, activation="softmax")])

    model.compile(optimizer="rmsprop", loss="SparseCategoricalCrossentropy", metrics=["accuracy"])

    model.summary()
    
    model.fit(train_ds, epochs=10,)


    return model




