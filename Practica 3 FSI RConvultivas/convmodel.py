# -*- coding: utf-8 -*-

# Sample code to use string producer.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """

    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h
    """
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3
batch_size = 4


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):


    #valor grande retrasará el inicio del entrenamiento,
    #  ya que TensorFlow tiene que procesar al menos tantos elementos antes de que el entrenamiento pueda comenzar.
    min_after_dequeue = 10

    # es un límite superior en la cantidad de memoria que consumirá la tubería de entrada:
    # establecer esto demasiado grande puede hacer que el proceso de capacitación se quede sin memoria
    # (y posiblemente comenzar a intercambiar, lo que perjudicará el rendimiento de la capacitación).
    capacity = min_after_dequeue + 3 * batch_size

    #lista ejemplos del lote (numeros que te van a dar las fotos)
    example_batch_list = []

    #lista etiquetas del lote (etiquetas que hacen en el one hot)
    label_batch_list = []

    # enumerar las rutas del arbol
    for i, p in enumerate(paths):

        # Aperturas del proyecto
        # train.match= Guarde la lista de archivos que coincidan con el patrón, por lo que solo se computa una vez.
        filename = tf.train.match_filenames_once(p)
        #Devuelve una cola con las cadenas de salida. Si es falso, las cadenas se barajan aleatoriamente
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        # Lector que devuelve todos los contenidos de un archivo como valor
        reader = tf.WholeFileReader()


        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), num_classes)

        #Decodificacion de imagenes
        #Cambia el tamaño de una imagen a un ancho y alto objetivo recortando de forma central la imagen o acolchándola uniformemente con ceros.
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)

        #Pasar a blanco y negro
        image=tf.image.rgb_to_grayscale(image)
        #Cambia la forma de un tensor. Devuelve un tensor que tiene los mismos valores que tensor en 3 dimensiones
        image = tf.reshape(image, [80, 140, 1])
        #Devuelve un tensor con la misma forma que image pero de tipo float (Casteo)
        image = tf.to_float(image) / 256. - 0.5

    #para optimizar las redes neuronales, se basan en registros de muestreo uniformemente al azar de todo el conjunto de entrenamiento.
    #  Sin embargo, no siempre es práctico cargar todo el conjunto de entrenamiento en la memoria (para tomar muestras de él),
    # por lo que tf.train.shuffle_batch()ofrece un compromiso: llena un búfer interno con elementos entre min_after_dequeuey capacity,
    # y da muestras uniformemente al azar de ese búfer.

        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        # Agrega a la lista
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    # Concatena los tensores
    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)
    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):

    # Crear las capas
    with tf.variable_scope('ConvNet', reuse=reuse):
        #conv2d: Construye una capa convolucional bidimensional.
        # Toma la imagen, el número de filtros, el tamaño del núcleo del filtro y la función de activación como argumentos.
        #Quita dos dimensiones

        #max_pooling2d: Construye una capa de puesta en común bidimensional utilizando el algoritmo de acumulación máxima.
        # Toma el tamaño del filtro de agrupación y camina como argumentos.
        #Elige 4 elementos y coge el más grande

        #dense: Construye una capa densa
        # Toma el número de neuronas y la función de activación como argumentos.

        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)


        # units= numeros de salida (Elementos que vamos a usar)
        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * num_classes, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(
    ['Frutas/0/*.jpg', 'Frutas/1/*.jpg', 'Frutas/2/*.jpg'],
    batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(
    ['Frutas/0_Validacion/*.jpg', 'Frutas/1_Validacion/*.jpg', 'Frutas/2_Validacion/*.jpg'],
    batch_size=batch_size)

example_batch_test, label_batch_test = dataSource(
    ['Frutas/0_Test/*.jpg', 'Frutas/1_Test/*.jpg', 'Frutas/2_Test/*.jpg'],
    batch_size=batch_size)

#False al principio para saber que es la primera vez y True para controlar el cambio
example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

# Calculamos el coste de la suma de los cuadrados
cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train,dtype=tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid,dtype=tf.float32)))
cost_test = tf.reduce_sum(tf.square(example_batch_test_predicted - tf.cast(label_batch_test,dtype=tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))

# Tenga en cuenta que la optimización no se realiza en este momento.
# Simplemente agregamos el objeto optimizador al gráfico TensorFlow para su posterior ejecución.
# El ratio cuanto más pequeño mas lento aprende pero es mas seguro para ver la grafica
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Guarda y restaura las variables

saver = tf.train.Saver()
grafictrain=[]
graficvalid=[]

with tf.Session() as sess:

    #La FileWriterclase proporciona un mecanismo para crear un archivo de eventos en un directorio determinado y agregarle resúmenes y eventos
    #Añade fichero al log
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    #Esta clase implementa un mecanismo simple para coordinar la terminación de un conjunto de hilos.
    coord = tf.train.Coordinator()
    #Inicia todos los corredores de colas recogidos en el gráfico.
    # sess: Session utilizado para ejecutar las operaciones de cola. Se establece de manera predeterminada en la sesión predeterminada.
    # coord: Opcional Coordinator para coordinar los hilos iniciados.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    Rango=600
    for _ in range(Rango):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            #print(sess.run(label_batch_valid))
            #print(sess.run(example_batch_valid_predicted))

            print("Error:", sess.run(cost_valid))

            # Coste del entreno para la grafica
            grafictrain.append(sess.run(cost))


            # Coste de la validacion para la grafica
            graficvalid.append(sess.run(cost_valid))
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)

    #Mostrar la grafica del entreno

    plt.title("Entrenamiento")
    plt.plot(grafictrain)
    plt.show()

    # Mostrar la grafica de validacion

    plt.title("Validacion")
    plt.plot(graficvalid)
    plt.show()


