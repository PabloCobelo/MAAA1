data = [1,2,3,4,5,6,7]
m = 1
normalized_data .=  (data - minimum(data)) / (maximum(data) - minimum(data) + m) 
print(normalized_data)