using FileIO
using JLD2
using Images
using FilePathsBase
using Flux
using Statistics
indexOutputLayer(ann::Chain) = length(ann) - (ann[end]==softmax);

function trainClassANN!(ann::Chain, trainingDataset::Tuple{AbstractArray{<:Real,2},
    AbstractArray{Bool,2}}, trainOnly2LastLayers::Bool;
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001,
    minLossChange::Real=1e-7, lossChangeWindowSize::Int=5) 

    inputs, targets = trainingDataset

    # Definimos la función de pérdida (loss)
    loss(model, x, y) = (size(y, 1) == 1) ? Flux.binarycrossentropy(model(x), y) : Flux.crossentropy(model(x), y)

    # Vectores para almacenar los valores de loss en cada ciclo
    trainingLosses = Float32[]
    
    # Cálculo inicial del loss antes de empezar el entrenamiento (ciclo 0)
    trainingLoss = loss(ann, inputs, targets)
    push!(trainingLosses, trainingLoss)
    # Configuramos el optimizador
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Si se especifica que sólo se entrenan las dos últimas capas
    if trainOnly2LastLayers
        Flux.freeze!(opt_state.layers[1:(indexOutputLayer(ann)-2)])
    end

    # Bucle de entrenamiento con las condiciones de parada
    numEpoch = 0
    while (numEpoch < maxEpochs) && (trainingLoss > minLoss)
        # Entrenamos un ciclo
        Flux.train!(loss, ann, [(inputs, targets)], opt_state)

        # Aumentamos el número de ciclos (épocas)
        numEpoch += 1

        # Calculamos el loss actual
        trainingLoss = loss(ann, inputs, targets)
        push!(trainingLosses, trainingLoss)

        # Comprobación del cambio mínimo en loss después de los ciclos indicados
        if numEpoch > lossChangeWindowSize
            lossWindow = trainingLosses[end-lossChangeWindowSize+1:end]
            minLossValue, maxLossValue = extrema(lossWindow)
            if (maxLossValue - minLossValue) / minLossValue <= minLossChange
                println("Early stopping: minimal change in loss detected.")
                break
            end
        end
    end

    return trainingLosses
end;
function newClassCascadeNetwork(numInputs::Int, numOutputs::Int)
    if numOutputs == 2
        # Para clasificación binaria (dos clases)
        return Chain(Dense(numInputs, 2, σ))
    elseif numOutputs > 2
        # Para clasificación multiclase (más de dos clases)
        return Chain(Dense(numInputs, numOutputs, identity), softmax)
    else
        # Para una sola salida
        return Chain(Dense(numInputs, 1, σ))
    end
end;
indexOutputLayer(ann::Chain) = length(ann) - (ann[end]==softmax);

function addClassCascadeNeuron(previousANN::Chain; transferFunction::Function=σ)
    # Obtener la capa de salida y las capas previas
    outputLayer = previousANN[indexOutputLayer(previousANN)];
    previousLayers = previousANN[1:(indexOutputLayer(previousANN)-1)];

    # Número de entradas y salidas de la capa de salida actual
    numInputsOutputLayer = size(outputLayer.weight, 2);
    numOutputsOutputLayer = size(outputLayer.weight, 1); 

    # Crear la nueva red con una neurona extra en cascada
    ann = Chain(
        previousLayers...,  # Expandir capas previas
        SkipConnection(Dense(numInputsOutputLayer, 1, transferFunction), (mx, x) -> vcat(x, mx)),
        # Capas de salida según el tipo de problema de clasificación
        numOutputsOutputLayer == 1 ? Dense(numInputsOutputLayer + 1, 1, σ) : Chain(Dense(numInputsOutputLayer + 1, numOutputsOutputLayer, identity), softmax)
    );

    # Modificar los pesos y bias de la nueva capa de salida
    # Crear matriz de pesos con una columna adicional
    newWeights = zeros(numOutputsOutputLayer, numInputsOutputLayer + 1);  
    # Copiar pesos antiguos a la nueva matriz, excepto la última columna
    newWeights[:, 1:numInputsOutputLayer] .= outputLayer.weight;
    # La última columna se pone a 0 para no afectar la salida
    newWeights[:, end] .= 0;  

    # El vector de bias permanece igual
    newBias = outputLayer.bias;

    if numOutputsOutputLayer == 1
        ann[end].weight .= newWeights;
        ann[end].bias .= newBias;
    else
        ann[end-1].weight .= newWeights;
        ann[end-1].bias .= newBias;
    end

    return ann  # Devolver la nueva red con la neurona añadida
end; 


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunction::Function=σ,
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
     
   
    inputs = Float32.(trainingDataset[1]')
    targets = trainingDataset[2]'
    
    #Crear RNA sin capas ocultas + entrenarla
    num_inputs = size(trainingDataset[1],2)
    num_outputs = length(unique(trainingDataset[2]))
    RNA = newClassCascadeNetwork(num_inputs,num_outputs)
 
    loss = []
 
 
    loss = trainClassANN!(RNA,(inputs,targets),false;maxEpochs=maxEpochs,minLoss=minLoss,learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)
 
    #Bucle entrenamiento
    
    for _ in 1:maxNumNeurons
 
        RNA = addClassCascadeNeuron(RNA;transferFunction = transferFunction)
        #Si el numero de neuronas es mayor que  1, congelamos las dos ultimas
        if length(RNA.layers) > 1 
            new_loss = trainClassANN!(RNA,(inputs,targets),true;maxEpochs=maxEpochs,minLoss=minLoss,learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)
            
            loss = vcat(loss,new_loss[2:end])
        end;
 
        #Volvemos a entrenar sin congelar
        new_loss = trainClassANN!(RNA,(inputs,targets),false;maxEpochs=maxEpochs,minLoss=minLoss,learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)
        loss = vcat(loss,new_loss[2:end])
    end;
     
    return RNA, loss
     
end;

function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunction::Function=σ,
    maxEpochs::Int=100, minLoss::Real=0.0, learningRate::Real=0.01, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    
    trainingDataset = (trainingDataset[1], reshape(trainingDataset[2], :, 2))

    return trainClassCascadeANN(maxNumNeurons, trainingDataset; transferFunction = transferFunction,maxEpochs=maxEpochs,minLoss=minLoss,learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)
end;


# Definir una red neuronal simple
ann = Chain(
    Dense(10, 5, relu),
    Dense(5, 2),
    softmax
)

# Crear un conjunto de datos de entrenamiento de ejemplo
X = rand(Int64, 368, 22)  # 368 muestras, 22 características
Y = rand(Bool,368*2)
transferFunction = σ    
maxEpochs = 1000
minLoss = 0.0
learningRate = 0.001
minLossChange = 1e-7
lossChangeWindowSize = 5
# Llamar a la función de entrenamiento
result = trainClassCascadeANN(5, (X, Y); transferFunction=transferFunction, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, minLossChange=minLossChange, lossChangeWindowSize=lossChangeWindowSize)

# Imprimir el historial de pérdida
println(result)