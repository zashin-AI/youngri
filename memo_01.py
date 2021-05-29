# accuracy
train_sizes, train_scores_model, test_scores_model = \
    learning_curve(model, x_train[:100], y_train[:100], train_sizes=np.linspace(0.1, 1.0, 10),
                   scoring="accuracy", cv=8, shuffle=True, random_state=42)

train_scores_mean = np.mean(train_scores_model, axis=1)
train_scores_std = np.std(train_scores_model, axis=1)
test_scores_mean = np.mean(test_scores_model, axis=1)
test_scores_std = np.std(test_scores_model, axis=1)

# log loss
train_sizes, train_scores_model, test_scores_model = \
    learning_curve(model, x_train[:100], y_train[:100], train_sizes=np.linspace(0.1, 1.0, 10),
                   scoring='neg_log_loss', cv=8, shuffle=True, random_state=42)

# accuracy
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="validation score")

# log loss
plt.plot(train_sizes, -train_scores_model.mean(1), 'o-', color="r", label="log_loss")
plt.plot(train_sizes, -test_scores_model.mean(1), 'o-', color="g", label="val log_loss")

plt.xlabel("Train size")
# plt.ylabel("Log loss")
plt.ylabel("Accuracy")
plt.title('lgbm')
plt.legend(loc="best")

plt.show()