
Generalized World Coordinate System
===================================

Nadia Dencheva
--------------

Perry Greenfield, Mike Droetboom
--------------------------------

Python in Astronomy Workshop
''''''''''''''''''''''''''''

Leiden 2015
'''''''''''

Goals
-----

-  Include all transformations from detector to a standard coordinate
   system

-  Combine transforms in an efficient way such that resampling is done
   as little as possible

Flexible
~~~~~~~~

-  Provide modular tools for managing WCS

-  combine transforms arbitrarily

-  execute subtransforms and their inverse

-  insert transforms in the pipeline

Extensible
~~~~~~~~~~

-  Uses astropy.modeling

-  Any other python executable can be used

-  Uses the astropy.coordinates framework

Combined models in astropy.modeling since v 1.0
-----------------------------------------------

.. code:: python

    from astropy.modeling.models import Mapping, Identity, Shift, Scale, Polynomial1D, Polynomial2D
    shift = Shift(1)
    scale = Scale(2)
    poly = Polynomial1D(1, c0=1.2, c1=2.3)

-  Binary arithmetic operations with models

.. code:: python

      model = (shift + poly) * scale / poly
      print model(1)


.. parsed-literal::

    3.14285714286
    

-  model composition - the output of model1 is passed as input to model2

.. code:: python

      model = shift | poly
      print model(1)


.. parsed-literal::

    5.8
    

-  model concatenation

.. code:: python

      model = Shift(1) & Shift(2)
      print model(1, 1)


.. parsed-literal::

    (2.0, 3.0)
    

-  axes management

-  modeling.models.Mapping
-  modeling.models.Identity

.. code:: python

        x, y = (1, 1)
        poly_x = Polynomial2D(degree=1, c0_0=1, c0_1=2, c1_0=2.1)
        poly_y = Polynomial2D(degree=4, c0_0=5, c1_0=1.2, c0_1=2)
        poly = poly_x & poly_y
        
        mapping = models.Mapping((0, 1, 0, 1))
        print("mapping.n_inputs:", mapping.n_inputs)
        print("mapping.n_ouputs:", mapping.n_outputs)
        
        model = mapping | poly
        print(model(x, y))
        


.. parsed-literal::

    ('mapping.n_inputs:', 2)
    ('mapping.n_ouputs:', 4)
    (5.1, 8.2)
    

.. code:: python

        mapping = models.Mapping((0,), n_inputs=2)
        print mapping(1, 2)
        


.. parsed-literal::

    1.0
    

.. code:: python

        model = Shift(1.2) & Identity(1)
        print model(1, 2)


.. parsed-literal::

    (2.2, 2.0)
    

This is a work in progress

-  Coordinate frames need to be fully developed
-  Create tools around the basic functionality

Source : https://github.com/spacetelescope/gwcs

Documentation: http://gwcs.readthedocs.org/en/latest/

