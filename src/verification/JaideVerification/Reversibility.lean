import JaideVerification.Scalar

set_option autoImplicit false

universe u v

namespace Jaide

variable {α : Type u} [RSFField α]

structure LayerParams (α : Type u) [RSFField α] where
  scaleFn : α → α
  transFn : α → α
  scaleFn_ne_zero : ∀ z : α, scaleFn z ≠ RSFField.zero

def couplingForward (p : LayerParams α) (x : α × α) : α × α :=
  (x.1 * p.scaleFn x.2, x.2 + p.transFn (x.1 * p.scaleFn x.2))

def couplingInverse (p : LayerParams α) (y : α × α) : α × α :=
  (y.1 / p.scaleFn (y.2 - p.transFn y.1), y.2 - p.transFn y.1)

def butterflyForward (y : α × α) : α × α :=
  ((y.1 - y.2) * RSFField.scale, (y.1 + y.2) * RSFField.scale)

def butterflyInverse (z : α × α) : α × α :=
  ((z.1 + z.2) * RSFField.scale, (z.2 - z.1) * RSFField.scale)

def layerForward (p : LayerParams α) (x : α × α) : α × α :=
  butterflyForward (couplingForward p x)

def layerInverse (p : LayerParams α) (y : α × α) : α × α :=
  couplingInverse p (butterflyInverse y)

def vectorForward (p : LayerParams α) (xs : List (α × α)) : List (α × α) :=
  xs.map (layerForward p)

def vectorInverse (p : LayerParams α) (xs : List (α × α)) : List (α × α) :=
  xs.map (layerInverse p)

def stackForward : List (LayerParams α) → List (α × α) → List (α × α)
  | [], xs => xs
  | p :: rest, xs => stackForward rest (vectorForward p xs)

def stackInverse : List (LayerParams α) → List (α × α) → List (α × α)
  | [], xs => xs
  | p :: rest, xs => vectorInverse p (stackInverse rest xs)

theorem coupling_left_inverse (p : LayerParams α) (x : α × α) :
    couplingInverse p (couplingForward p x) = x :=
  match x with
  | (x1, x2) =>
    have h_arg :
        x2 + p.transFn (x1 * p.scaleFn x2) - p.transFn (x1 * p.scaleFn x2) = x2 :=
      RSFField.add_sub_cancel x2 (p.transFn (x1 * p.scaleFn x2))
    have h_scale_arg :
        p.scaleFn (x2 + p.transFn (x1 * p.scaleFn x2) - p.transFn (x1 * p.scaleFn x2)) =
          p.scaleFn x2 :=
      congrArg p.scaleFn h_arg
    have h_fst1 :
        x1 * p.scaleFn x2 /
            p.scaleFn (x2 + p.transFn (x1 * p.scaleFn x2) - p.transFn (x1 * p.scaleFn x2)) =
          x1 * p.scaleFn x2 / p.scaleFn x2 :=
      congrArg (fun w : α => x1 * p.scaleFn x2 / w) h_scale_arg
    have h_ne : p.scaleFn x2 ≠ RSFField.zero :=
      p.scaleFn_ne_zero x2
    have h_fst2 : x1 * p.scaleFn x2 / p.scaleFn x2 = x1 :=
      RSFField.mul_div_cancel x1 (p.scaleFn x2) h_ne
    have h_fst :
        x1 * p.scaleFn x2 /
            p.scaleFn (x2 + p.transFn (x1 * p.scaleFn x2) - p.transFn (x1 * p.scaleFn x2)) =
          x1 :=
      Eq.trans h_fst1 h_fst2
    have h_snd :
        x2 + p.transFn (x1 * p.scaleFn x2) - p.transFn (x1 * p.scaleFn x2) = x2 :=
      h_arg
    Prod.ext h_fst h_snd

theorem coupling_right_inverse (p : LayerParams α) (y : α × α) :
    couplingForward p (couplingInverse p y) = y :=
  match y with
  | (y1, y2) =>
    have h_ne : p.scaleFn (y2 - p.transFn y1) ≠ RSFField.zero :=
      p.scaleFn_ne_zero (y2 - p.transFn y1)
    have h_div :
        y1 / p.scaleFn (y2 - p.transFn y1) * p.scaleFn (y2 - p.transFn y1) = y1 :=
      RSFField.div_mul_cancel y1 (p.scaleFn (y2 - p.transFn y1)) h_ne
    have h_trans_arg :
        p.transFn (y1 / p.scaleFn (y2 - p.transFn y1) * p.scaleFn (y2 - p.transFn y1)) =
          p.transFn y1 :=
      congrArg p.transFn h_div
    have h_snd1 :
        y2 - p.transFn y1 +
            p.transFn (y1 / p.scaleFn (y2 - p.transFn y1) * p.scaleFn (y2 - p.transFn y1)) =
          y2 - p.transFn y1 + p.transFn y1 :=
      congrArg (fun w : α => y2 - p.transFn y1 + w) h_trans_arg
    have h_snd2 : y2 - p.transFn y1 + p.transFn y1 = y2 :=
      RSFField.sub_add_cancel y2 (p.transFn y1)
    have h_snd :
        y2 - p.transFn y1 +
            p.transFn (y1 / p.scaleFn (y2 - p.transFn y1) * p.scaleFn (y2 - p.transFn y1)) =
          y2 :=
      Eq.trans h_snd1 h_snd2
    have h_fst :
        y1 / p.scaleFn (y2 - p.transFn y1) * p.scaleFn (y2 - p.transFn y1) = y1 :=
      h_div
    Prod.ext h_fst h_snd

theorem butterfly_left_inverse (y : α × α) :
    butterflyInverse (butterflyForward y) = y :=
  match y with
  | (a, b) =>
    have h_sum1 :
        (a - b) * RSFField.scale + (a + b) * RSFField.scale =
          (a - b + (a + b)) * RSFField.scale :=
      Eq.symm (RSFField.add_mul (a - b) (a + b) RSFField.scale)
    have h_sum2 : a - b + (a + b) = a + a :=
      RSFField.sub_add_self a b
    have h_sum3 :
        (a - b + (a + b)) * RSFField.scale = (a + a) * RSFField.scale :=
      congrArg (fun t : α => t * RSFField.scale) h_sum2
    have h_sum :
        (a - b) * RSFField.scale + (a + b) * RSFField.scale = (a + a) * RSFField.scale :=
      Eq.trans h_sum1 h_sum3
    have h_fst1 :
        ((a - b) * RSFField.scale + (a + b) * RSFField.scale) * RSFField.scale =
          (a + a) * RSFField.scale * RSFField.scale :=
      congrArg (fun t : α => t * RSFField.scale) h_sum
    have h_fst2 :
        (a + a) * RSFField.scale * RSFField.scale =
          (a + a) * (RSFField.scale * RSFField.scale) :=
      RSFField.mul_assoc (a + a) RSFField.scale RSFField.scale
    have h_fst3 : (a + a) * (RSFField.scale * RSFField.scale) = a :=
      Eq.trans
        (congrArg (fun s : α => (a + a) * s) RSFField.scale_sq)
        (RSFField.double_half a)
    have h_fst :
        ((a - b) * RSFField.scale + (a + b) * RSFField.scale) * RSFField.scale = a :=
      Eq.trans h_fst1 (Eq.trans h_fst2 h_fst3)
    have h_diff1 :
        (a + b) * RSFField.scale - (a - b) * RSFField.scale =
          (a + b - (a - b)) * RSFField.scale :=
      Eq.symm (RSFField.sub_mul (a + b) (a - b) RSFField.scale)
    have h_diff2 : a + b - (a - b) = b + b :=
      RSFField.add_sub_self a b
    have h_diff3 :
        (a + b - (a - b)) * RSFField.scale = (b + b) * RSFField.scale :=
      congrArg (fun t : α => t * RSFField.scale) h_diff2
    have h_diff :
        (a + b) * RSFField.scale - (a - b) * RSFField.scale = (b + b) * RSFField.scale :=
      Eq.trans h_diff1 h_diff3
    have h_snd1 :
        ((a + b) * RSFField.scale - (a - b) * RSFField.scale) * RSFField.scale =
          (b + b) * RSFField.scale * RSFField.scale :=
      congrArg (fun t : α => t * RSFField.scale) h_diff
    have h_snd2 :
        (b + b) * RSFField.scale * RSFField.scale =
          (b + b) * (RSFField.scale * RSFField.scale) :=
      RSFField.mul_assoc (b + b) RSFField.scale RSFField.scale
    have h_snd3 : (b + b) * (RSFField.scale * RSFField.scale) = b :=
      Eq.trans
        (congrArg (fun s : α => (b + b) * s) RSFField.scale_sq)
        (RSFField.double_half b)
    have h_snd :
        ((a + b) * RSFField.scale - (a - b) * RSFField.scale) * RSFField.scale = b :=
      Eq.trans h_snd1 (Eq.trans h_snd2 h_snd3)
    Prod.ext h_fst h_snd

theorem butterfly_right_inverse (z : α × α) :
    butterflyForward (butterflyInverse z) = z :=
  match z with
  | (a, b) =>
    have h_diff1 :
        (a + b) * RSFField.scale - (b - a) * RSFField.scale =
          (a + b - (b - a)) * RSFField.scale :=
      Eq.symm (RSFField.sub_mul (a + b) (b - a) RSFField.scale)
    have h_diff2 : a + b - (b - a) = a + a :=
      RSFField.add_sub_swap a b
    have h_diff3 :
        (a + b - (b - a)) * RSFField.scale = (a + a) * RSFField.scale :=
      congrArg (fun t : α => t * RSFField.scale) h_diff2
    have h_diff :
        (a + b) * RSFField.scale - (b - a) * RSFField.scale = (a + a) * RSFField.scale :=
      Eq.trans h_diff1 h_diff3
    have h_fst1 :
        ((a + b) * RSFField.scale - (b - a) * RSFField.scale) * RSFField.scale =
          (a + a) * RSFField.scale * RSFField.scale :=
      congrArg (fun t : α => t * RSFField.scale) h_diff
    have h_fst2 :
        (a + a) * RSFField.scale * RSFField.scale =
          (a + a) * (RSFField.scale * RSFField.scale) :=
      RSFField.mul_assoc (a + a) RSFField.scale RSFField.scale
    have h_fst3 : (a + a) * (RSFField.scale * RSFField.scale) = a :=
      Eq.trans
        (congrArg (fun s : α => (a + a) * s) RSFField.scale_sq)
        (RSFField.double_half a)
    have h_fst :
        ((a + b) * RSFField.scale - (b - a) * RSFField.scale) * RSFField.scale = a :=
      Eq.trans h_fst1 (Eq.trans h_fst2 h_fst3)
    have h_sum1 :
        (a + b) * RSFField.scale + (b - a) * RSFField.scale =
          (a + b + (b - a)) * RSFField.scale :=
      Eq.symm (RSFField.add_mul (a + b) (b - a) RSFField.scale)
    have h_sum2 : a + b + (b - a) = b + b :=
      RSFField.add_add_self a b
    have h_sum3 :
        (a + b + (b - a)) * RSFField.scale = (b + b) * RSFField.scale :=
      congrArg (fun t : α => t * RSFField.scale) h_sum2
    have h_sum :
        (a + b) * RSFField.scale + (b - a) * RSFField.scale = (b + b) * RSFField.scale :=
      Eq.trans h_sum1 h_sum3
    have h_snd1 :
        ((a + b) * RSFField.scale + (b - a) * RSFField.scale) * RSFField.scale =
          (b + b) * RSFField.scale * RSFField.scale :=
      congrArg (fun t : α => t * RSFField.scale) h_sum
    have h_snd2 :
        (b + b) * RSFField.scale * RSFField.scale =
          (b + b) * (RSFField.scale * RSFField.scale) :=
      RSFField.mul_assoc (b + b) RSFField.scale RSFField.scale
    have h_snd3 : (b + b) * (RSFField.scale * RSFField.scale) = b :=
      Eq.trans
        (congrArg (fun s : α => (b + b) * s) RSFField.scale_sq)
        (RSFField.double_half b)
    have h_snd :
        ((a + b) * RSFField.scale + (b - a) * RSFField.scale) * RSFField.scale = b :=
      Eq.trans h_snd1 (Eq.trans h_snd2 h_snd3)
    Prod.ext h_fst h_snd

theorem layer_left_inverse (p : LayerParams α) (x : α × α) :
    layerInverse p (layerForward p x) = x :=
  have h1 :
      butterflyInverse (butterflyForward (couplingForward p x)) = couplingForward p x :=
    butterfly_left_inverse (couplingForward p x)
  have h2 :
      couplingInverse p (butterflyInverse (butterflyForward (couplingForward p x))) =
        couplingInverse p (couplingForward p x) :=
    congrArg (couplingInverse p) h1
  have h3 : couplingInverse p (couplingForward p x) = x :=
    coupling_left_inverse p x
  have h_main :
      couplingInverse p (butterflyInverse (butterflyForward (couplingForward p x))) = x :=
    Eq.trans h2 h3
  h_main

theorem layer_right_inverse (p : LayerParams α) (y : α × α) :
    layerForward p (layerInverse p y) = y :=
  have h1 :
      couplingForward p (couplingInverse p (butterflyInverse y)) = butterflyInverse y :=
    coupling_right_inverse p (butterflyInverse y)
  have h2 :
      butterflyForward (couplingForward p (couplingInverse p (butterflyInverse y))) =
        butterflyForward (butterflyInverse y) :=
    congrArg butterflyForward h1
  have h3 : butterflyForward (butterflyInverse y) = y :=
    butterfly_right_inverse y
  have h_main :
      butterflyForward (couplingForward p (couplingInverse p (butterflyInverse y))) = y :=
    Eq.trans h2 h3
  h_main

theorem map_left_inverse {β : Type v}
    (f g : β → β) (h : ∀ x : β, g (f x) = x) :
    ∀ xs : List β, (xs.map f).map g = xs :=
  fun xs =>
    List.rec
      (motive := fun ys : List β => (ys.map f).map g = ys)
      (Eq.refl ([] : List β))
      (fun (a : β) (tail : List β) (ih : (tail.map f).map g = tail) =>
        have h1 :
            ((a :: tail).map f).map g = g (f a) :: (tail.map f).map g :=
          Eq.refl (g (f a) :: (tail.map f).map g)
        have h2 :
            g (f a) :: (tail.map f).map g = g (f a) :: tail :=
          congrArg (fun ys : List β => g (f a) :: ys) ih
        have h3 : g (f a) = a :=
          h a
        have h4 : g (f a) :: tail = a :: tail :=
          congrArg (fun x : β => x :: tail) h3
        have h5 : ((a :: tail).map f).map g = a :: tail :=
          Eq.trans h1 (Eq.trans h2 h4)
        h5)
      xs

theorem vector_left_inverse (p : LayerParams α) (xs : List (α × α)) :
    vectorInverse p (vectorForward p xs) = xs :=
  have h_pointwise : ∀ z : α × α, layerInverse p (layerForward p z) = z :=
    layer_left_inverse p
  have h_map :
      (xs.map (layerForward p)).map (layerInverse p) = xs :=
    map_left_inverse (layerForward p) (layerInverse p) h_pointwise xs
  h_map

theorem vector_right_inverse (p : LayerParams α) (xs : List (α × α)) :
    vectorForward p (vectorInverse p xs) = xs :=
  have h_pointwise : ∀ z : α × α, layerForward p (layerInverse p z) = z :=
    layer_right_inverse p
  have h_map :
      (xs.map (layerInverse p)).map (layerForward p) = xs :=
    map_left_inverse (layerInverse p) (layerForward p) h_pointwise xs
  h_map

theorem stack_left_inverse :
    ∀ (ps : List (LayerParams α)) (xs : List (α × α)),
      stackInverse ps (stackForward ps xs) = xs :=
  fun ps =>
    List.rec
      (motive := fun qs : List (LayerParams α) =>
        ∀ xs : List (α × α), stackInverse qs (stackForward qs xs) = xs)
      (fun xs => Eq.refl xs)
      (fun (p : LayerParams α) (rest : List (LayerParams α))
          (ih : ∀ xs : List (α × α), stackInverse rest (stackForward rest xs) = xs)
          (xs : List (α × α)) =>
        have h1 :
            stackInverse (p :: rest) (stackForward (p :: rest) xs) =
              vectorInverse p (stackInverse rest (stackForward rest (vectorForward p xs))) :=
          Eq.refl (vectorInverse p (stackInverse rest (stackForward rest (vectorForward p xs))))
        have h2 :
            stackInverse rest (stackForward rest (vectorForward p xs)) = vectorForward p xs :=
          ih (vectorForward p xs)
        have h3 :
            vectorInverse p (stackInverse rest (stackForward rest (vectorForward p xs))) =
              vectorInverse p (vectorForward p xs) :=
          congrArg (vectorInverse p) h2
        have h4 : vectorInverse p (vectorForward p xs) = xs :=
          vector_left_inverse p xs
        have h5 : stackInverse (p :: rest) (stackForward (p :: rest) xs) = xs :=
          Eq.trans h1 (Eq.trans h3 h4)
        h5)
      ps

theorem stack_right_inverse :
    ∀ (ps : List (LayerParams α)) (xs : List (α × α)),
      stackForward ps (stackInverse ps xs) = xs :=
  fun ps =>
    List.rec
      (motive := fun qs : List (LayerParams α) =>
        ∀ xs : List (α × α), stackForward qs (stackInverse qs xs) = xs)
      (fun xs => Eq.refl xs)
      (fun (p : LayerParams α) (rest : List (LayerParams α))
          (ih : ∀ xs : List (α × α), stackForward rest (stackInverse rest xs) = xs)
          (xs : List (α × α)) =>
        have h1 :
            stackForward (p :: rest) (stackInverse (p :: rest) xs) =
              stackForward rest (vectorForward p (vectorInverse p (stackInverse rest xs))) :=
          Eq.refl (stackForward rest (vectorForward p (vectorInverse p (stackInverse rest xs))))
        have h2 :
            vectorForward p (vectorInverse p (stackInverse rest xs)) = stackInverse rest xs :=
          vector_right_inverse p (stackInverse rest xs)
        have h3 :
            stackForward rest (vectorForward p (vectorInverse p (stackInverse rest xs))) =
              stackForward rest (stackInverse rest xs) :=
          congrArg (stackForward rest) h2
        have h4 : stackForward rest (stackInverse rest xs) = xs :=
          ih xs
        have h5 : stackForward (p :: rest) (stackInverse (p :: rest) xs) = xs :=
          Eq.trans h1 (Eq.trans h3 h4)
        h5)
      ps

theorem stackForward_injective
    (ps : List (LayerParams α))
    (xs ys : List (α × α))
    (h : stackForward ps xs = stackForward ps ys) :
    xs = ys :=
  have h1 : xs = stackInverse ps (stackForward ps xs) :=
    Eq.symm (stack_left_inverse ps xs)
  have h2 :
      stackInverse ps (stackForward ps xs) = stackInverse ps (stackForward ps ys) :=
    congrArg (stackInverse ps) h
  have h3 : stackInverse ps (stackForward ps ys) = ys :=
    stack_left_inverse ps ys
  have h_main : xs = ys :=
    Eq.trans h1 (Eq.trans h2 h3)
  h_main

theorem stackInverse_injective
    (ps : List (LayerParams α))
    (xs ys : List (α × α))
    (h : stackInverse ps xs = stackInverse ps ys) :
    xs = ys :=
  have h1 : xs = stackForward ps (stackInverse ps xs) :=
    Eq.symm (stack_right_inverse ps xs)
  have h2 :
      stackForward ps (stackInverse ps xs) = stackForward ps (stackInverse ps ys) :=
    congrArg (stackForward ps) h
  have h3 : stackForward ps (stackInverse ps ys) = ys :=
    stack_right_inverse ps ys
  have h_main : xs = ys :=
    Eq.trans h1 (Eq.trans h2 h3)
  h_main

end Jaide
