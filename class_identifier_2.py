import os

folders = []
avg_discs = []
rankings = ["S+", "S", "S-","A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F+", "F", "F-"]
rank_eval = []
directory = "/Users/brandon/practice-SPIS/dataset"
keyword = "label"

def calculate_weighted_avg(discrepancy):
    perfects = []
    alrights = []
    mids = []
    for value in discrepancy:
        if value < 0.25:
            perfects.append(value)
        elif value < 5.0:
            alrights.append(value)
        else:
            mids.append(value)

    return (sum(perfects)+sum(alrights)+sum(mids))/len(discrepancy)
                

def find_avg_disc(filename):
    # Grabbing the first and third columns (indexes 0 and 2)
    column_1 = []
    column_3 = []
    discrepancy = []

    with open(filename, "r") as file:
        for line in file:
            # split() automatically handles spaces and tabs, while strip() removes whitespace at the ends
            parts = line.strip().split()
            
            # Ensure the line isn't empty and has enough columns
            if len(parts) > 3:
                column_1.append(parts[1])
                column_3.append(parts[3])
            
    for i in range(len(column_1)):
        discrepancy.append(abs(round((float(column_1[i]) - float(column_3[i])), 2)))

    return calculate_weighted_avg(discrepancy)


for root, _, filenames in os.walk(directory):
    for filename in filenames:
        if keyword in filename:
            text_filename = os.path.join(root, filename)
            folders.append(text_filename)
            avg_discs.append(find_avg_disc(text_filename))

print(avg_discs)

for index, discrepancy in enumerate(avg_discs):
    func = 0
    rank_index = 0
    while discrepancy >= func and rank_index < len(rankings) - 1:
        func += 0.25
        rank_index += 1
    rank_eval.append(rankings[rank_index])
    
print(rank_eval)

    


