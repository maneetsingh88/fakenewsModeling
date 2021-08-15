from mesa import Agent, Model
from mesa.time import RandomActivation
import networkx as nx
from mesa.space import NetworkGrid
import enum
import random
import numpy as np
from mesa.datacollection import DataCollector
import pandas as pd


class MyAgent(Agent):
    
    
    #Intializing agent attributes.
    def __init__(self, unique_id, influence,initial_opinion,group_support,status,originator,model):
        
        super().__init__(unique_id, model)
        self.id=unique_id
        self.opinion=round(initial_opinion,2)
        self.influence=influence
        self.group_support=group_support
        self.status=status
        self.likelihood_convincing=0
        self.extremeness=abs(2*self.opinion-1)
        self.originator=originator
        self.assessed_neighbors=[]
            
    def step(self):
        
        # An agent's step
        neighbors=self.model.grid.get_neighbors(self.pos, include_center=False)
        eventp = np.random.random_sample()
        
        if self.status == 0:  # Susceptible
            
            infected_neighbors = [v for v in neighbors if self.model.schedule.agents[v].status == 1]
            if len(infected_neighbors)>0:
                
                #Status Update
                self.status = 2 #Exposed
                self.model.num_exposed+=1
                self.model.num_susceptible-=1
                self.model.num_consumers+=1
                
                
                #Spread inspection and Opinion Update
                for neigh in infected_neighbors:
                    nb_opinion=self.model.opinions[neigh]
                    avg_opinion=(nb_opinion+self.model.schedule.agents[self.model.schedule.agents[neigh].originator].opinion)/2
                    org_opinion=self.model.schedule.agents[self.model.schedule.agents[neigh].originator].opinion
                    if abs(nb_opinion-self.opinion)<=self.model.op_eps:

                        conf=conf=self.model.weight+(1-self.model.weight)*self.extremeness
                        delta_opinion=nb_opinion-self.opinion
                        self.opinion=self.opinion+(1-conf)*delta_opinion
                        self.extremeness=abs(2*self.opinion-1)
                        
                        if abs(nb_opinion-self.opinion)<=self.model.op_eps:
                            if self.status!=1:       #Not already infected
                                sinf=self.model.schedule.agents[neigh].influence
                                ext=self.extremeness
                                ech=self.group_support
                                alpha=(sinf+ext+ech)/3
                                if eventp<alpha:
                                    self.status = 1  # Infected
                                    self.model.num_infected+=1
                                    self.model.num_exposed-=1
                                    self.originator=self.model.schedule.agents[neigh].originator
                                    self.model.num_spreaders+=1
                
                        
                self.assessed_neighbors.extend(infected_neighbors)        
                    
                        
                        
        elif self.status == 2:
            #Finding alpha each time and checking for infecting probability each time
            infected_neighbors = [v for v in neighbors if self.model.schedule.agents[v].status == 1]
            to_be_assessed_neighbors=list(set(infected_neighbors).difference(self.assessed_neighbors))
            if len(to_be_assessed_neighbors)>0:
                
                
                #Spread inspection and Opinion Update
                for neigh in to_be_assessed_neighbors:
                    
                    if neigh not in self.assessed_neighbors:
                        self.assessed_neighbors.append(neigh)
                    nb_opinion=self.model.opinions[neigh]
                    avg_opinion=(nb_opinion+self.model.schedule.agents[self.model.schedule.agents[neigh].originator].opinion)/2
                    org_opinion=self.model.schedule.agents[self.model.schedule.agents[neigh].originator].opinion
                    if abs(nb_opinion-self.opinion)<=self.model.op_eps:

                        conf=conf=self.model.weight+(1-self.model.weight)*self.extremeness
                        delta_opinion=nb_opinion-self.opinion
                        self.opinion=self.opinion+(1-conf)*delta_opinion
                        self.extremeness=abs(2*self.opinion-1)
                        
                        if abs(nb_opinion-self.opinion)<=self.model.op_eps:
                            if self.status!=1:       #Not already infected
                                sinf=self.model.schedule.agents[neigh].influence
                                ext=self.extremeness
                                ech=self.group_support
                                alpha=(sinf+ext+ech)/3
                                if eventp<alpha:
                                    self.status = 1  # Infected
                                    self.model.num_infected+=1
                                    self.model.num_exposed-=1
                                    self.originator=self.model.schedule.agents[neigh].originator
                                    self.model.num_spreaders+=1
                self.assessed_neighbors.extend(to_be_assessed_neighbors)
                         
        elif self.status == 1:
            
            if eventp < self.model.beta:
                self.status = 3  # Removed
                self.model.num_recovered+=1
                self.model.num_infected-=1
        
        pass
            
class hybridmodeling(Model):
    #Initializing model attributes
    def __init__(self,G,epsilon,beta,weight):
        self.num_agents = G.number_of_nodes()
        self.grid = NetworkGrid(G)
        self.schedule = RandomActivation(self)
        self.flag=0
        self.op_eps=epsilon
        self.beta=beta
        self.weight=weight
        influence=nx.get_node_attributes(G,'influence')
        initial_opinion=nx.get_node_attributes(G,'initial_opinion')
        group_support=nx.get_node_attributes(G,'group_support')
        status=nx.get_node_attributes(G,'status')
        originator=nx.get_node_attributes(G,'originator')
        status_list=list(status.values())
        self.num_exposed=status_list.count(2)
        self.num_infected=status_list.count(1)
        self.num_recovered=status_list.count(3)
        self.num_susceptible=status_list.count(0)
        self.num_consumers=self.num_infected
        self.num_spreaders=self.num_infected
        self.num_leftist=len([op for op in list(initial_opinion.values()) if op<=0.4])
        self.num_rightist=len([op for op in list(initial_opinion.values()) if op>=0.6])
        self.opinions=initial_opinion
        num_trials=5
        self.num_exposed_queue=[-i for i in range(num_trials-1)]+[self.num_exposed]
        self.num_infected_queue=[-i for i in range(num_trials-1)]+[self.num_infected]
        self.num_recovered_queue=[-i for i in range(num_trials-1)]+[self.num_recovered]
        self.num_susceptible_queue=[-i for i in range(num_trials-1)]+[self.num_susceptible]
        # Create the agents
        for i, node in enumerate(G.nodes()):
            a = MyAgent(i, influence[node],initial_opinion[node],group_support[node],status[node],originator[node],self)
            self.schedule.add(a)
            self.grid.place_agent(a, node)
        self.datacollector = DataCollector(model_reporters={"Number of Consumers":"num_consumers","Number of Spreaders":"num_spreaders","Exposed":"num_exposed","Infected":"num_infected","Recovered":"num_recovered","Susceptible":"num_susceptible","Number of Leftist":"num_leftist","Number of Rightist":"num_rightist"},agent_reporters={"status":"status","opinion": "opinion","extremeness":"extremeness","influence":"influence"})
   

    def step(self):
        #A model's step
        self.datacollector.collect(self)
        
        self.schedule.step()
        self.num_exposed_queue.pop(0)
        self.num_infected_queue.pop(0)
        self.num_recovered_queue.pop(0)
        self.num_susceptible_queue.pop(0)
        self.num_exposed_queue.append(self.num_exposed)
        self.num_infected_queue.append(self.num_infected)
        self.num_recovered_queue.append(self.num_recovered)
        self.num_susceptible_queue.append(self.num_susceptible)
        prev_opinion=self.opinions.copy()
        self.opinions=[(self.schedule.agents[nd]).opinion for nd in range(self.num_agents)]
        self.num_leftist=len([op for op in self.opinions if op<=0.4])
        self.num_rightist=len([op for op in self.opinions if op>=0.6])
            
        
        if len(set(self.num_exposed_queue))==1 and len(set(self.num_infected_queue))==1 and len(set(self.num_recovered_queue))==1 and len(set(self.num_susceptible_queue))==1 and prev_opinion==self.opinions:
            self.flag=1
        else:
            self.flag=0
       
