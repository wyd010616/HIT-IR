class LR_b():
    def _prepare(self, df_train):
        self.tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        train_data = self.tv.fit_transform(df_train['sent'])
        return train_data
    
    def fit(self, df_train):
        x_train = self._prepare(df_train)
        y_train = df_train['Blabel']
        lr = LogisticRegression(max_iter=400, n_jobs=-1)
        param_grid = [{'C': [100, 500, 1000, 5000]}]
        self.model = GridSearchCV(lr, param_grid, cv=3, n_jobs=-1).fit(x_train, y_train)
        
    def predict(self, df_test):
        x_test = self.tv.transform(df_test['sent'])
        y_pre = self.model.predict(x_test)
        return y_pre
    
# 小类
class LR_s():
    def _prepare(self, df_train):
        self.tv = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        train_data = self.tv.fit_transform(df_train['sent'])
        return train_data
    
    def fit(self, df_train):
        x_train = self._prepare(df_train)
        y_train = df_train['label']
        lr = LogisticRegression(C=500)
        self.model = lr.fit(x_train, y_train)
        
    def predict(self, df_test):
        x_test = self.tv.transform(df_test['sent'])
        y_pre = self.model.predict(x_test)
        return y_pre