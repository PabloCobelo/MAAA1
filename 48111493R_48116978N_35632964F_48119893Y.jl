# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using FileIO
using JLD2
using Images
using FilePathsBase
using Flux

function fileNamesFolder(folderName::String, extension::String)
    # Verificamos si el nombre de carpeta es válido
    if !isdir(folderName)
        error("El nombre de carpeta NO es valido.")
    end

    # Convertir la extensión a mayúsculas
    extension = uppercase(extension)
    
    # Obtener los nombres de los archivos que terminan con la extensión dada
    fileNames = filter(f -> endswith(uppercase(f), ".$extension"), readdir(folderName))
    
    # Eliminar la extensión de los nombres de archivo
    fileNamesWithoutExtension = [splitext(f)[1] for f in fileNames]
    
    return fileNamesWithoutExtension
end;
#println(typeof(fileNamesFolder("datasets", "tsv")), " ", fileNamesFolder("datasets", "tsv"))


using DelimitedFiles

    function loadDataset(datasetName::String, datasetFolder::String; datasetType::DataType=Float32)
        # Construir el nombre completo del archivo
        fileName = joinpath(datasetFolder, datasetName * ".tsv")
    
        # Verificar si el archivo existe
        if !isfile(fileName)
            return nothing  # Si no se encuentra el archivo, retornar nothing
        end
    
        # Cargar el archivo delimitado por tabulaciones
        dataset = readdlm(fileName, '\t')
    
        # Obtener los encabezados (primera fila)
        encabezados = dataset[1, :]
    
        # Encontrar la columna correspondiente al "target"
        target_col = findall(x -> x == "target", encabezados)[1]
    
        # Extraer las entradas, excluyendo la columna del "target"
        inputs = dataset[2:end, setdiff(1:end, target_col)]
        inputs = convert(Matrix{datasetType}, inputs)
    
        # Extraer la columna de "targets" y convertirla a booleanos
        targets = dataset[2:end, target_col]
        targets = convert(Vector{Bool}, targets .== 1)
    
        # Retornar las entradas y las salidas
        return (inputs, targets)
    end;


    function loadImage(imageName::String, datasetFolder::String; datasetType::DataType=Float32, resolution::Int=128)
        # Construir el nombre completo del archivo
        imagePath = joinpath(datasetFolder, imageName * ".tif")
        
        # Verificar si el archivo existe
        if !isfile(imagePath)
            return nothing  # Si no se encuentra el archivo, retornar nothing
        end
    
        # Cargar la imagen
        image = load(imagePath)
        
        # Redimensionar la imagen
        image_resized = imresize(image, (resolution, resolution))
    
        # Convertir los píxeles de la imagen a escala de grises y luego al tipo numérico especificado
        image_gray = Gray.(image_resized)
        image_converted = map(x -> convert(datasetType, float(x)), image_gray)
    
        return image_converted
    end;

function convertImagesNCHW(imageVector::Vector{<:AbstractArray{<:Real,2}})
    imagesNCHW = Array{eltype(imageVector[1]), 4}(undef, length(imageVector), 1, size(imageVector[1],1), size(imageVector[1],2));
    for numImage in Base.OneTo(length(imageVector))
        imagesNCHW[numImage,1,:,:] .= imageVector[numImage];
    end;
    return imagesNCHW;
end;


function loadImagesNCHW(datasetFolder::String; datasetType::DataType=Float32, resolution::Int=128)
    image = fileNamesFolder(datasetFolder, "tif");
    # Hacerun broadcast de la función loadImage, (aplicar una función a cada elemento de un array)
    images = loadImage.(image, Ref(datasetFolder); datasetType=datasetType, resolution=resolution)
    
    # Convertir las imágenes al formato NCHW
    nchw_images = convertImagesNCHW(images)
    
    # Devolver las imágenes en formato NCHW
    return nchw_images
end;


showImage(image      ::AbstractArray{<:Real,2}                                      ) = display(Gray.(image));
showImage(imagesNCHW ::AbstractArray{<:Real,4}                                      ) = display(Gray.(     hcat([imagesNCHW[ i,1,:,:] for i in 1:size(imagesNCHW ,1)]...)));
showImage(imagesNCHW1::AbstractArray{<:Real,4}, imagesNCHW2::AbstractArray{<:Real,4}) = display(Gray.(vcat(hcat([imagesNCHW1[i,1,:,:] for i in 1:size(imagesNCHW1,1)]...), hcat([imagesNCHW2[i,1,:,:] for i in 1:size(imagesNCHW2,1)]...))));





function loadMNISTDataset(datasetFolder::String; labels::AbstractArray{Int,1}=0:9, datasetType::DataType=Float32)
    # Cargar el dataset completo desde el archivo MNIST.jld2
    dataset_file = joinpath(datasetFolder, "MNIST.jld2")
    dataset = load(dataset_file)

    # Extraer los datos de entrenamiento y test
    train_imgs = dataset["train_imgs"]
    train_labels = dataset["train_labels"]
    test_imgs = dataset["test_imgs"]
    test_labels = dataset["test_labels"]
    
    # Modificar los targets para etiquetas no contempladas en labels
    train_labels[.!in.(train_labels, [setdiff(labels, -1)])] .= -1
    test_labels[.!in.(test_labels, [setdiff(labels, -1)])] .= -1

    # Seleccionar las imágenes y etiquetas deseadas
    train_indices = in.(train_labels, [labels])
    test_indices = in.(test_labels, [labels])

    # Convertir las imágenes a formato NCHW (esto lo implementas en tu propia función)
    train_imgs_nchw = convertImagesNCHW(train_imgs[train_indices])
    test_imgs_nchw = convertImagesNCHW(test_imgs[test_indices])

    # Filtrar las etiquetas correspondientes
    train_labels_filtered = train_labels[train_indices]
    test_labels_filtered = test_labels[test_indices]

    # Convertir las imágenes a tipo datasetType (Float32 por defecto)
    train_imgs_nchw = convert(Array{datasetType, 4}, train_imgs_nchw)
    test_imgs_nchw = convert(Array{datasetType, 4}, test_imgs_nchw)

    # Devolver la tupla con las imágenes 4D y las etiquetas
    return (train_imgs_nchw, train_labels_filtered, test_imgs_nchw, test_labels_filtered)
end;



function intervalDiscreteVector(data::AbstractArray{<:Real,1})
    # Ordenar los datos
    uniqueData = sort(unique(data));
    # Obtener diferencias entre elementos consecutivos
    differences = sort(diff(uniqueData));
    # Tomar la diferencia menor
    minDifference = differences[1];
    # Si todas las diferencias son multiplos exactos (valores enteros) de esa diferencia, entonces es un vector de valores discretos
    isInteger(x::Float64, tol::Float64) = abs(round(x)-x) < tol
    return all(isInteger.(differences./minDifference, 1e-3)) ? minDifference : 0.
end;


function cyclicalEncoding(data::AbstractArray{<:Real,1})
    #Obtener valor de m
    m = intervalDiscreteVector(data)

    #Obtener valores minimo y maximo de los datos para normalizar
    maximo = maximum(data)
    minimo = minimum(data)

    #Normalizar los datos
    normalized_data = (data .- minimo) ./ (maximo - minimo + m)

    #Mapear a intervalo [0,2pi]
    angles = normalized_data .* (2*pi)

    #Calcular senos y cosenos

    sin_values = sin.(angles)
    cos_values = cos.(angles)

    # Devolver los resultados como una tupla (senos, cosenos)
    return (sin_values, cos_values)
end;


function loadStreamLearningDataset(datasetFolder::String; datasetType::DataType=Float32)
    # Cargar los archivos de datos y etiquetas
    data_file = joinpath(datasetFolder, "elec2_data.dat")
    label_file = joinpath(datasetFolder, "elec2_label.dat")
    
    # Leer los datos
    data = readdlm(data_file)
    labels = readdlm(label_file)
    
    # Eliminar las columnas 1 (date) y 4 (nswprice)
    data_processed = data[:, setdiff(1:size(data, 2), [1, 4])]
    
    # Aplicar cyclicalEncoding a la primera columna (period)
    sin_vals, cos_vals = cyclicalEncoding(data_processed[:, 1])
    
    # Concatenar senos y cosenos como las primeras columnas
    inputs = hcat(sin_vals, cos_vals, data_processed[:, 2:end])
    
    # Convertir los datos al tipo especificado (datasetType)
    inputs_converted = convert(Matrix{datasetType}, inputs)
    
    # Convertir las etiquetas a booleano
    targets = vec(Bool.(labels))
    
    # Devolver la matriz de entradas y el vector de salidas deseadas
    return (inputs_converted, targets)
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------


indexOutputLayer(ann::Chain) = length(ann) - (ann[end]==softmax);



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

function trainClassANN!(ann::Chain, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, trainOnly2LastLayers::Bool;
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)

    # Preparar el dataset
    X, y = trainingDataset
    X = Float32.(X)
    y = Float32.(y)

    # Definir la función de pérdida y el optimizador
    loss(x, y) = Flux.logitcrossentropy(ann(x), y)
    opt = ADAM(learningRate)

    # Crear el vector para almacenar el historial de pérdida
    loss_history = Float32[]

    # Función para registrar la pérdida
    function log_loss()
        push!(loss_history, loss(X, y))
    end

    # Inicializar el historial de pérdida
    log_loss()

    # Bucle de entrenamiento
    for epoch in 1:maxEpochs
        # Si es necesario, congelar todas las capas menos las dos últimas
        if trainOnly2LastLayers
            # Freezing the first layers by setting their gradients to zero
            for p in Flux.params(ann[1:(end-2)])
                p.grad .= 0
            end
        end

        # Entrenar una época completa
        Flux.train!(loss, Flux.params(ann), [(X, y)], opt)

        # Registrar la pérdida
        log_loss()

        # Chequeo de criterios de parada temprana
        if length(loss_history) > lossChangeWindowSize
            if abs(loss_history[end] - loss_history[end - lossChangeWindowSize]) < minLossChange
                println("Deteniendo entrenamiento: cambio mínimo en loss")
                break
            end
        end

        if loss_history[end] < minLoss
            println("Deteniendo entrenamiento: loss mínima alcanzada")
            break
        end
    end

    return loss_history
end  


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunction::Function=σ,
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
     
    #Trasponer matrices de entrada
    trainingDataset = (trainingDataset[1]',trainingDataset[2]')

    #Crear RNA sin capas ocultas + entrenarla
    num_inputs = size(trainingDataset[1],2)
    RNA = newClassCascadeNetwork(num_inputs,2)
 
    loss = Float32[]
 
    loss = trainClassANN!(RNA,trainingDataset,true,maxEpochs,minLoss,learningRate, minLossChange, lossChangeWindowSize)
 
    #Bucle entrenamiento
    #Si fallan valores cambiar 1 por 2
    for numNeurons in 1:maxNumNeurons
 
        RNA = addClassCascadeNeuron(RNA)
        #Si el numero de neuronas es mayor que  1, congelamos las dos ultimas
        if numNeurons > 1 
            loss = [loss;trainClassANN!(RNA,trainingDataset,true,maxEpochs,minLoss,learningRate, minLossChange, lossChangeWindowSize )[2:end]]
            
        end;
 
        #Volvemos a entrenar sin congelar
        loss = [loss;trainClassANN!(RNA,trainingDataset,true,maxEpochs,minLoss,learningRate, minLossChange, lossChangeWindowSize)[2:end]]
 
    end;
     
    return RNA, loss
     
end;

function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunction::Function=σ,
    maxEpochs::Int=100, minLoss::Real=0.0, learningRate::Real=0.01, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    
    trainingDataset = (trainingDataset[1], reshape(trainingDataset[2], :, 1))

    return trainClassCascadeANN(maxNumNeurons, trainingDataset, transferFunction, maxEpochs, minLoss, learningRate, minLossChange, lossChangeWindowSize)
end;
    

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

HopfieldNet = Array{Float32,2}

function trainHopfield(trainingSet::AbstractArray{<:Real,2})
    num_instances, num_atributes = size(trainingSet)
    matriz = zeros(Float32, num_atributes, num_atributes)
    matriz .= (trainingSet' * trainingSet) ./ num_instances
    for i in 1:num_atributes
        matriz[i, i] = 0.0
    end
    return matriz
end;
function trainHopfield(trainingSet::AbstractArray{<:Bool,2})
    convertido = (2. .*trainingSet) .- 1 
    return trainHopfield(convertido)
end;
function trainHopfield(trainingSetNCHW::AbstractArray{<:Bool,4})
    convertido2 = reshape(trainingSetNCHW, size(trainingSetNCHW, 1), size(trainingSetNCHW, 2) * size(trainingSetNCHW, 3) * size(trainingSetNCHW, 4))
    return trainHopfield(convertido2)
end;


function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    S_float = Float32.(S)
    siguiente = ann * S_float
    final = sign.(siguiente)
    return Float32.(final)
end;
function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Bool,1})
    convertido = (2. .*S) .- 1
    resultado_real = stepHopfield(ann, convertido)
    return resultado_real .>= 0
end;


function runHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    prev_S = nothing;
    prev_prev_S = nothing;
    while S!=prev_S && S!=prev_prev_S
        prev_prev_S = prev_S;
        prev_S = S;
        S = stepHopfield(ann, S);
    end;
    return S
end;
function runHopfield(ann::HopfieldNet, dataset::AbstractArray{<:Real,2})
    outputs = copy(dataset);
    for i in 1:size(dataset,1)
        outputs[i,:] .= runHopfield(ann, view(dataset, i, :));
    end;
    return outputs;
end;
function runHopfield(ann::HopfieldNet, datasetNCHW::AbstractArray{<:Real,4})
    outputs = runHopfield(ann, reshape(datasetNCHW, size(datasetNCHW,1), size(datasetNCHW,3)*size(datasetNCHW,4)));
    return reshape(outputs, size(datasetNCHW,1), 1, size(datasetNCHW,3), size(datasetNCHW,4));
end;


using Random
function addNoise(datasetNCHW::AbstractArray{<:Bool,4}, ratioNoise::Real)
    noiseSet = copy(datasetNCHW)
    # Seleccionar los índices de las imágenes a modificar
    indices = shuffle(1:length(noiseSet))[1:Int(round(length(noiseSet)*ratioNoise))];
    noiseSet[indices] .= .!noiseSet[indices]
    return noiseSet
end;

function cropImages(datasetNCHW::AbstractArray{<:Bool,4}, ratioCrop::Real)
    cropSet = copy(datasetNCHW)
    # Obtener las dimensiones del dataset
    N, C, H, W = size(cropSet)
    colu_crop = round(Int(W * ratioCrop)) # imagina 8 x 0.25 --> se recortan 2 columnas
    cropSet[:, :, :, (W - colu_crop + 1):W] .= false #selcciona desde la columna W - colu_crop + 1 hasta W y las pone en false
    return cropSet
end;


function randomImages(numImages::Int, resolution::Int)
    images = randn( numImages, 1, resolution, resolution)
    return images .>= 0
end;

function averageMNISTImages(imageArray::AbstractArray{<:Real,4}, labelArray::AbstractArray{Int,1})
    labels = unique(labelArray) 
    outputImages = (imageArray, length(labels), size(imageArray, 2), size(imageArray, 3), size(imageArray, 4))
    for indexLabel in labels
        outputImages[indexLabel, :, :, :] = dropdims(mean(imageArray[labelArray .== indexLabel, 1, :, :], dims=1), dims=1)
    end
    return outputImages, labels

end;

function classifyMNISTImages(imageArray::AbstractArray{<:Real,4}, templateInputs::AbstractArray{<:Real,4}, templateLabels::AbstractArray{Int,1})
    #
    # Codigo a desarrollar
    #
end;

function calculateMNISTAccuracies(datasetFolder::String, labels::AbstractArray{Int,1}, threshold::Real)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


using ScikitLearn: @sk_import, fit!, predict
@sk_import svm: SVC

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function batchInputs(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchTargets(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchLength(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function selectInstances(batch::Batch, indices::Any)
    #
    # Codigo a desarrollar
    #
end;

function joinBatches(batch1::Batch, batch2::Batch)
    #
    # Codigo a desarrollar
    #
end;


function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
    #
    # Codigo a desarrollar
    #
end;

function addBatch!(memory::Batch, newBatch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function euclideanDistances(memory::Batch, instance::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(memory::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(memory::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_KNN(datasetFolder::String, windowSize::Int, batchSize::Int, k::Int)
    #
    # Codigo a desarrollar
    #
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function predictKNN_SVM(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int, C::Real)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN_SVM(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int, C::Real)
    #
    # Codigo a desarrollar
    #
end;
