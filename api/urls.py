from django.urls import path
from .views import AskPdView, AskPdWalkerView, AskGraphView, AskGraphStepsView, PromptPdView, PromptGraphView, LoginView

urlpatterns = [
    path('login', LoginView.as_view(), name='login'),
    path('ask/pd', AskPdView.as_view(), name='ask_pd'),
    path('ask/pd-walker', AskPdWalkerView.as_view(), name='ask_pd_walker'),
    path('ask/graph', AskGraphView.as_view(), name='ask_graph'),
    path('ask/graph-steps', AskGraphStepsView.as_view(), name='ask_graph_steps'),
    path('prompt/pd', PromptPdView.as_view(), name='prompt_pd'),
    path('prompt/graph', PromptGraphView.as_view(), name='prompt_graph'),
]