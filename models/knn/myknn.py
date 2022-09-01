import numpy as np
import statistics as st

class myKNN(object):

    """Classe que retorna predições do KNN
    """

    def __init__(self, k, types_):
        
        self.k = k
        self.types_= types_

    def dist_euclidean(self, p, data):
        """Calcula a distância euclidiana"""
        return np.sqrt(np.sum((p - data)**2, axis=1))


    def common_elements(self, list_):
        """Seleciona o número máximo em uma lista"""
        return max(set(list_), key=list_.count)

    def fitModel(self,  X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def pred_(self, X_test):

        """Retorna as previsões com base nos dados de teste"""
        
        ngb = []

        for x in X_test:
            dist = self.dist_euclidean(x, self.X_train)
            y_sorting = [y for _, y in sorted(zip(dist, self.y_train))]
            ngb.append(y_sorting[:self.k])

        return list(map(self.common_elements, ngb))

    def eval_(self, X_test, y_test):

        """
        Retorna os dados de avaliação do modelo com base na seleção do tipo, se regressão ou classificação
        
        """
        
        if self.types_ == 'reg':
            y_pred = self.pred_(X_test)
            print(st.mean(y_pred))

            accuracy = sum(y_pred == y_test) / len(y_test)
        
        elif self.types_ == 'cla':
          y_pred = self.pred_(X_test)
          print(st.mode(y_pred))

          accuracy = sum(y_pred == y_test) / len(y_test)

                
        return accuracy