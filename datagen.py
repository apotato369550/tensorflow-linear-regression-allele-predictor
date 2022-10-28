import random
import csv
import io

'''
this program helps create training and evaluation data for our machine learning program
The goal: Predicting the child's dominant hand given the alleles of both parents, together with their dominant hands, and the child's alleles
The Method: Punnett Squares, TensorFlow, and Python/JuPyter

Quick Summary: Data categories are as follows:

p1_alleles,p2_alleles,p1_dominant,p2_dominant,child_alleles,child_dominant
aa,AA,0,1,Aa,1

p1_alleles - Alleles of the first parent
p2_alleles - Alleles of the second parent
p1_dominant - Dominant hand of the first parent
p2_dominant - Dominant hand of the second parent
child_alleles - Alleles of the child
child_dominant - Dominant hand of the child

Alleles and their traits:
(A) Dominant Allele - Right-handedness
(a) Recessive Allele - Left-handedness

Allele Combinations:
(AA) Homozygous dominant - In here, the dominant trait is expressed. In this scenario, the parent is right handed and carries
the right-handed trait to their offspring
(Aa) Heterozygous dominant - In here, the dominant trait is also expressed. In this scenario, the parent is also right handed,
but carries the dominant right-handed allele and the recessive left-handed allele which may be passed on to their offspring
(aa) Homozygous Recessive - In here, the recessive trait is expressed. The parent is left handed and carries the recessive trait
to their offspring

> The alleles of the parents are randomly generated. 
> Their dominant hand is calculated randomly based on the following rules:
    > If the parent is homo-dominant - The parent is right handed
    > If the parent is hetero-dominant - The parent has a 50/50 chance to be left or right handed
    > If the parent is homo-recessive - The parent is left handed
> The dominant hand of the child is generated based off the alleles of the parents and the probability of
being left or right handed when the allele pairs are placed in a punnet square:
    > If both parents are homo-dominant - The child is right handed and homo-dominant
    > If one parent is 

# continue this
# fix this
'''

def main():
    with io.open("train.csv", "w", newline="") as file:
        train_writer = csv.writer(file)

        train_writer.writerow(["p1_alleles", "p2_alleles", "p1_dominant", "p2_dominant", "child_alleles", "child_dominant"])
     
        for i in range(500):
            p1_alleles = "A" if random.randint(0, 1) == 0 else "a"
            if p1_alleles == "a":
                p1_alleles += "a"
            else:
                p1_alleles += "A" if random.randint(0, 1) == 0 else "a"

            p2_alleles = "A" if random.randint(0, 1) == 0 else "a"
            if p2_alleles == "a":
                p2_alleles += "a"
            else:
                p2_alleles += "A" if random.randint(0, 1) == 0 else "a"
            
            p1_dominant = 1 if p1_alleles[0] == "A" else 0
            p2_dominant = 1 if p2_alleles[0] == "A" else 0

            possible_child_alleles = [
                str(p1_alleles[0] + p2_alleles[0]), 
                str(p1_alleles[1] + p2_alleles[1]), 
                str(p1_alleles[0] + p2_alleles[1]), 
                str(p1_alleles[1] + p2_alleles[0]),
            ]

            for j in range(len(possible_child_alleles)):
                if possible_child_alleles[j][0] == "a" and possible_child_alleles[j][1] == "A":
                    possible_child_alleles[j] = possible_child_alleles[j][::-1]




            child_alleles = possible_child_alleles[random.randint(0, 3)]
            child_dominant = 1 if child_alleles[0] == "A" else 0

            train_writer.writerow([p1_alleles, p2_alleles, p1_dominant, p2_dominant, child_alleles, child_dominant])

        with io.open("eval.csv", "w", newline="") as file:
            eval_writer = csv.writer(file)

            eval_writer.writerow(["p1_alleles", "p2_alleles", "p1_dominant", "p2_dominant", "child_alleles", "child_dominant"])
        
            for i in range(200):
                p1_alleles = "A" if random.randint(0, 1) == 0 else "a"
                if p1_alleles == "a":
                    p1_alleles += "a"
                else:
                    p1_alleles += "A" if random.randint(0, 1) == 0 else "a"

                p2_alleles = "A" if random.randint(0, 1) == 0 else "a"
                if p2_alleles == "a":
                    p2_alleles += "a"
                else:
                    p2_alleles += "A" if random.randint(0, 1) == 0 else "a"
                
                p1_dominant = 1 if p1_alleles[0] == "A" else 0
                p2_dominant = 1 if p2_alleles[0] == "A" else 0

                possible_child_alleles = [
                    str(p1_alleles[0] + p2_alleles[0]), 
                    str(p1_alleles[1] + p2_alleles[1]), 
                    str(p1_alleles[0] + p2_alleles[1]), 
                    str(p1_alleles[1] + p2_alleles[0]),
                ]

                for j in range(len(possible_child_alleles)):
                    if possible_child_alleles[j][0] == "a" and possible_child_alleles[j][1] == "A":
                        possible_child_alleles[j] = possible_child_alleles[j][::-1]




                child_alleles = possible_child_alleles[random.randint(0, 3)]
                child_dominant = 1 if child_alleles[0] == "A" else 0

                eval_writer.writerow([p1_alleles, p2_alleles, p1_dominant, p2_dominant, child_alleles, child_dominant])

            
    return


if __name__ == "__main__":
    main()