from KNN import KNN

def main():    
    X = [[1,0,'B'], [1,2, 'B'], [2,2,'A'], [2,3,'B'], [3,2,'A']]
    P = [1,2]
    K = 2
    print("X : {}".format(str(X)))
    print("P : {}".format(str(P)))
    #distance - 'euclidean', 'manhattan', 'hamming' 
    clf = KNN(X, K, distance='manhattan',weighted=True)
    print("Calculating K: {} nearest neighbors for P ...".format(K))
    output = clf.query(P)
    print(output)    
        
main()