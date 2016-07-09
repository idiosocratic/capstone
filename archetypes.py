import random




def get_archetypes(number_of_archs, some_memory):
    
    assert not number_of_archs == 0
    assert not len(some_memory) == 0
    
    archetypes = []

    for i in range(number_of_archs):
    
        archetypes.append(random.choice(some_memory))
        
    return archetypes    
        
        
        

def rebalance_archetypes(old_archetypes, updated_memory):

    number_of_archs = len(old_archetypes)
    
    archetype_child_lists = [] # list of lists for states belonging to each archetype
    
    for num in range(number_of_archs):
    
        archetype_child_lists.append([])  # initialize lists for each archetype
    
    new_archetypes = []  # list for rebalanced archetypes
    
    for state in updated_memory:
    
        closest_arch = None # initialize index as invalid
        
        closest_distance = 1000 # initialize as large number
        
        for index, arch in enumerate(old_archetypes):
        
            dist = np.linalg.norm(state-arch)  # get L2 distance between current state,arch pair
            
            if dist < closest_distance:
            
                closest_distance = dist
                
                closest_arch = index
                
        assert not closest_arch == None        
        
        archetype_child_lists[closest_arch].append(state) # assign state to closest arch child list
                
    for child_list in archetype_child_lists:
    
        new_archetype = sum(child_list)/float(len(child_list)) # get average of child_list            
                
        new_archetypes.append(new_archetype)
        
        
    return new_archetypes
    
    
    
    
# build-model, get transitions by checking if action changed archetype state  

# model-free, i.e. policy gradients: change "archetype action" to reflect average action of good episodes

#       
        
        