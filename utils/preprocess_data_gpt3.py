if __name__ == '__main__':
    with open("../data/corpus.en_ru.1m.en", "r") as f:
        trg = f.readlines()

    with open("../data/corpus.en_ru.1m.ru", "r") as f:
        src = f.readlines()

    general = [f"<s> {src_sent.rstrip()} <translation> {target.rstrip()}</s>\n" for src_sent, target in zip(src, trg)]
    general = general[:int(0.1*len(general))]

    open("general.ln", "w").writelines(general[:int(0.8*len(general))])
    open("general.valid.ln", "w").writelines(general[int(0.8 * len(general)):])