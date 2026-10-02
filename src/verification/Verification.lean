set_option autoImplicit false

universe u v

namespace Jaide

class RSFScalar (α : Type u) extends Add α, Sub α, Mul α, Div α where
  zero : α
  half : α
  fractalScale : α
  lt : α → α → Bool
  add_comm : ∀ a b : α, a + b = b + a
  add_assoc : ∀ a b c : α, a + b + c = a + (b + c)
  mul_assoc : ∀ a b c : α, a * b * c = a * (b * c)
  add_mul : ∀ a b c : α, (a + b) * c = a * c + b * c
  sub_mul : ∀ a b c : α, (a - b) * c = a * c - b * c
  add_sub : ∀ a b c : α, a + (b - c) = a + b - c
  sub_add : ∀ a b c : α, a - b + c = a + c - b
  sub_sub_eq_add : ∀ a b c : α, a - (b - c) = a - b + c
  add_sub_cancel : ∀ a b : α, a + b - b = a
  sub_add_cancel : ∀ a b : α, a - b + b = a
  fractalScale_sq : fractalScale * fractalScale = half
  double_half : ∀ a : α, (a + a) * half = a
  mul_div_cancel : ∀ a b : α, b ≠ zero → a * b / b = a
  div_mul_cancel : ∀ a b : α, b ≠ zero → a / b * b = a

namespace RSFScalar

variable {α : Type u} [RSFScalar α]

theorem sub_add_self (a b : α) : a - b + (a + b) = a + a :=
  calc a - b + (a + b)
      = a + (a + b) - b := RSFScalar.sub_add a b (a + b)
    _ = a + a + b - b := congrArg (fun t => t - b) (RSFScalar.add_assoc a a b).symm
    _ = a + a := RSFScalar.add_sub_cancel (a + a) b

theorem add_sub_self (a b : α) : a + b - (a - b) = b + b :=
  calc a + b - (a - b)
      = a + b - a + b := RSFScalar.sub_sub_eq_add (a + b) a b
    _ = b + a - a + b := congrArg (fun t => t - a + b) (RSFScalar.add_comm a b)
    _ = b + b := congrArg (fun t => t + b) (RSFScalar.add_sub_cancel b a)

theorem add_sub_swap (a b : α) : a + b - (b - a) = a + a :=
  calc a + b - (b - a)
      = a + b - b + a := RSFScalar.sub_sub_eq_add (a + b) b a
    _ = a + a := congrArg (fun t => t + a) (RSFScalar.add_sub_cancel a b)

theorem add_add_self (a b : α) : a + b + (b - a) = b + b :=
  calc a + b + (b - a)
      = a + b + b - a := RSFScalar.add_sub (a + b) b a
    _ = a + (b + b) - a := congrArg (fun t => t - a) (RSFScalar.add_assoc a b b)
    _ = b + b + a - a := congrArg (fun t => t - a) (RSFScalar.add_comm a (b + b))
    _ = b + b := RSFScalar.add_sub_cancel (b + b) a

instance unitModel : RSFScalar Unit where
  add _ _ := ()
  sub _ _ := ()
  mul _ _ := ()
  div _ _ := ()
  zero := ()
  half := ()
  fractalScale := ()
  lt _ _ := false
  add_comm _ _ := rfl
  add_assoc _ _ _ := rfl
  mul_assoc _ _ _ := rfl
  add_mul _ _ _ := rfl
  sub_mul _ _ _ := rfl
  add_sub _ _ _ := rfl
  sub_add _ _ _ := rfl
  sub_sub_eq_add _ _ _ := rfl
  add_sub_cancel _ _ := rfl
  sub_add_cancel _ _ := rfl
  fractalScale_sq := rfl
  double_half _ := rfl
  mul_div_cancel _ _ _ := rfl
  div_mul_cancel _ _ _ := rfl

end RSFScalar

variable {α : Type u} [RSFScalar α]

structure ElementWeights (α : Type u) [RSFScalar α] where
  sWeight : α
  sBias : α
  tWeight : α
  tBias : α
  clipMin : α
  clipMax : α
  expFn : α → α
  expFn_ne_zero : ∀ z : α, expFn z ≠ RSFScalar.zero

def clipTo (lo hi v : α) : α :=
  if RSFScalar.lt v lo then lo else if RSFScalar.lt hi v then hi else v

def scaleAt (w : ElementWeights α) (z : α) : α :=
  w.expFn (clipTo w.clipMin w.clipMax (w.sWeight * z + w.sBias))

def transAt (w : ElementWeights α) (z : α) : α :=
  w.tWeight * z + w.tBias

theorem scaleAt_ne_zero (w : ElementWeights α) (z : α) : scaleAt w z ≠ RSFScalar.zero :=
  w.expFn_ne_zero (clipTo w.clipMin w.clipMax (w.sWeight * z + w.sBias))

def couplingForwardPair (w : ElementWeights α) (x : α × α) : α × α :=
  (x.1 * scaleAt w x.2, x.2 + transAt w (x.1 * scaleAt w x.2))

def couplingInversePair (w : ElementWeights α) (y : α × α) : α × α :=
  (y.1 / scaleAt w (y.2 - transAt w y.1), y.2 - transAt w y.1)

def oftbForwardPair (x : α × α) : α × α :=
  ((x.1 - x.2) * RSFScalar.fractalScale, (x.1 + x.2) * RSFScalar.fractalScale)

def oftbInversePair (z : α × α) : α × α :=
  ((z.1 + z.2) * RSFScalar.fractalScale, (z.2 - z.1) * RSFScalar.fractalScale)

def couplingForwardRow : List (ElementWeights α) → List (α × α) → List (α × α)
  | [], xs => xs
  | _ :: _, [] => []
  | w :: ws, x :: xs => couplingForwardPair w x :: couplingForwardRow ws xs

def couplingInverseRow : List (ElementWeights α) → List (α × α) → List (α × α)
  | [], ys => ys
  | _ :: _, [] => []
  | w :: ws, y :: ys => couplingInversePair w y :: couplingInverseRow ws ys

def oftbForwardRow : List (α × α) → List (α × α)
  | [] => []
  | x :: xs => oftbForwardPair x :: oftbForwardRow xs

def oftbInverseRow : List (α × α) → List (α × α)
  | [] => []
  | z :: zs => oftbInversePair z :: oftbInverseRow zs

def layerForwardRow (ws : List (ElementWeights α)) (row : List (α × α)) : List (α × α) :=
  oftbForwardRow (couplingForwardRow ws row)

def layerInverseRow (ws : List (ElementWeights α)) (row : List (α × α)) : List (α × α) :=
  couplingInverseRow ws (oftbInverseRow row)

def layerForwardBatch (ws : List (ElementWeights α)) :
    List (List (α × α)) → List (List (α × α))
  | [] => []
  | row :: rows => layerForwardRow ws row :: layerForwardBatch ws rows

def layerInverseBatch (ws : List (ElementWeights α)) :
    List (List (α × α)) → List (List (α × α))
  | [] => []
  | row :: rows => layerInverseRow ws row :: layerInverseBatch ws rows

def stackForward : List (List (ElementWeights α)) →
    List (List (α × α)) → List (List (α × α))
  | [], batch => batch
  | ws :: rest, batch => stackForward rest (layerForwardBatch ws batch)

def stackInverse : List (List (ElementWeights α)) →
    List (List (α × α)) → List (List (α × α))
  | [], batch => batch
  | ws :: rest, batch => layerInverseBatch ws (stackInverse rest batch)

theorem couplingPair_left_inverse (w : ElementWeights α) (x : α × α) :
    couplingInversePair w (couplingForwardPair w x) = x :=
  match x with
  | (x1, x2) =>
    have h_arg :
        x2 + transAt w (x1 * scaleAt w x2) - transAt w (x1 * scaleAt w x2) = x2 :=
      RSFScalar.add_sub_cancel x2 (transAt w (x1 * scaleAt w x2))
    have h_scale_arg :
        scaleAt w (x2 + transAt w (x1 * scaleAt w x2) - transAt w (x1 * scaleAt w x2)) =
          scaleAt w x2 :=
      congrArg (scaleAt w) h_arg
    have h_fst1 :
        x1 * scaleAt w x2 /
            scaleAt w (x2 + transAt w (x1 * scaleAt w x2) - transAt w (x1 * scaleAt w x2)) =
          x1 * scaleAt w x2 / scaleAt w x2 :=
      congrArg (fun t : α => x1 * scaleAt w x2 / t) h_scale_arg
    have h_fst2 : x1 * scaleAt w x2 / scaleAt w x2 = x1 :=
      RSFScalar.mul_div_cancel x1 (scaleAt w x2) (scaleAt_ne_zero w x2)
    have h_fst :
        x1 * scaleAt w x2 /
            scaleAt w (x2 + transAt w (x1 * scaleAt w x2) - transAt w (x1 * scaleAt w x2)) =
          x1 :=
      Eq.trans h_fst1 h_fst2
    Prod.ext h_fst h_arg

theorem couplingPair_right_inverse (w : ElementWeights α) (y : α × α) :
    couplingForwardPair w (couplingInversePair w y) = y :=
  match y with
  | (y1, y2) =>
    have h_div :
        y1 / scaleAt w (y2 - transAt w y1) * scaleAt w (y2 - transAt w y1) = y1 :=
      RSFScalar.div_mul_cancel y1 (scaleAt w (y2 - transAt w y1))
        (scaleAt_ne_zero w (y2 - transAt w y1))
    have h_trans_arg :
        transAt w (y1 / scaleAt w (y2 - transAt w y1) * scaleAt w (y2 - transAt w y1)) =
          transAt w y1 :=
      congrArg (transAt w) h_div
    have h_snd1 :
        y2 - transAt w y1 +
            transAt w (y1 / scaleAt w (y2 - transAt w y1) * scaleAt w (y2 - transAt w y1)) =
          y2 - transAt w y1 + transAt w y1 :=
      congrArg (fun t : α => y2 - transAt w y1 + t) h_trans_arg
    have h_snd2 : y2 - transAt w y1 + transAt w y1 = y2 :=
      RSFScalar.sub_add_cancel y2 (transAt w y1)
    have h_snd :
        y2 - transAt w y1 +
            transAt w (y1 / scaleAt w (y2 - transAt w y1) * scaleAt w (y2 - transAt w y1)) =
          y2 :=
      Eq.trans h_snd1 h_snd2
    Prod.ext h_div h_snd

theorem oftbPair_left_inverse (x : α × α) :
    oftbInversePair (oftbForwardPair x) = x :=
  match x with
  | (a, b) =>
    have h_sum :
        (a - b) * RSFScalar.fractalScale + (a + b) * RSFScalar.fractalScale =
          (a + a) * RSFScalar.fractalScale :=
      Eq.trans
        (Eq.symm (RSFScalar.add_mul (a - b) (a + b) RSFScalar.fractalScale))
        (congrArg (fun t : α => t * RSFScalar.fractalScale) (RSFScalar.sub_add_self a b))
    have h_fst :
        ((a - b) * RSFScalar.fractalScale + (a + b) * RSFScalar.fractalScale) *
            RSFScalar.fractalScale = a :=
      Eq.trans
        (congrArg (fun t : α => t * RSFScalar.fractalScale) h_sum)
        (Eq.trans
          (RSFScalar.mul_assoc (a + a) RSFScalar.fractalScale RSFScalar.fractalScale)
          (Eq.trans
            (congrArg (fun t : α => (a + a) * t) RSFScalar.fractalScale_sq)
            (RSFScalar.double_half a)))
    have h_diff :
        (a + b) * RSFScalar.fractalScale - (a - b) * RSFScalar.fractalScale =
          (b + b) * RSFScalar.fractalScale :=
      Eq.trans
        (Eq.symm (RSFScalar.sub_mul (a + b) (a - b) RSFScalar.fractalScale))
        (congrArg (fun t : α => t * RSFScalar.fractalScale) (RSFScalar.add_sub_self a b))
    have h_snd :
        ((a + b) * RSFScalar.fractalScale - (a - b) * RSFScalar.fractalScale) *
            RSFScalar.fractalScale = b :=
      Eq.trans
        (congrArg (fun t : α => t * RSFScalar.fractalScale) h_diff)
        (Eq.trans
          (RSFScalar.mul_assoc (b + b) RSFScalar.fractalScale RSFScalar.fractalScale)
          (Eq.trans
            (congrArg (fun t : α => (b + b) * t) RSFScalar.fractalScale_sq)
            (RSFScalar.double_half b)))
    Prod.ext h_fst h_snd

theorem oftbPair_right_inverse (z : α × α) :
    oftbForwardPair (oftbInversePair z) = z :=
  match z with
  | (a, b) =>
    have h_diff :
        (a + b) * RSFScalar.fractalScale - (b - a) * RSFScalar.fractalScale =
          (a + a) * RSFScalar.fractalScale :=
      Eq.trans
        (Eq.symm (RSFScalar.sub_mul (a + b) (b - a) RSFScalar.fractalScale))
        (congrArg (fun t : α => t * RSFScalar.fractalScale) (RSFScalar.add_sub_swap a b))
    have h_fst :
        ((a + b) * RSFScalar.fractalScale - (b - a) * RSFScalar.fractalScale) *
            RSFScalar.fractalScale = a :=
      Eq.trans
        (congrArg (fun t : α => t * RSFScalar.fractalScale) h_diff)
        (Eq.trans
          (RSFScalar.mul_assoc (a + a) RSFScalar.fractalScale RSFScalar.fractalScale)
          (Eq.trans
            (congrArg (fun t : α => (a + a) * t) RSFScalar.fractalScale_sq)
            (RSFScalar.double_half a)))
    have h_sum :
        (a + b) * RSFScalar.fractalScale + (b - a) * RSFScalar.fractalScale =
          (b + b) * RSFScalar.fractalScale :=
      Eq.trans
        (Eq.symm (RSFScalar.add_mul (a + b) (b - a) RSFScalar.fractalScale))
        (congrArg (fun t : α => t * RSFScalar.fractalScale) (RSFScalar.add_add_self a b))
    have h_snd :
        ((a + b) * RSFScalar.fractalScale + (b - a) * RSFScalar.fractalScale) *
            RSFScalar.fractalScale = b :=
      Eq.trans
        (congrArg (fun t : α => t * RSFScalar.fractalScale) h_sum)
        (Eq.trans
          (RSFScalar.mul_assoc (b + b) RSFScalar.fractalScale RSFScalar.fractalScale)
          (Eq.trans
            (congrArg (fun t : α => (b + b) * t) RSFScalar.fractalScale_sq)
            (RSFScalar.double_half b)))
    Prod.ext h_fst h_snd

theorem couplingRow_left_inverse :
    ∀ (ws : List (ElementWeights α)) (row : List (α × α)),
      couplingInverseRow ws (couplingForwardRow ws row) = row :=
  fun ws =>
    List.rec
      (motive := fun qs : List (ElementWeights α) =>
        ∀ row : List (α × α), couplingInverseRow qs (couplingForwardRow qs row) = row)
      (fun row => Eq.refl row)
      (fun (w : ElementWeights α) (rest : List (ElementWeights α))
          (ih : ∀ row : List (α × α), couplingInverseRow rest (couplingForwardRow rest row) = row)
          (row : List (α × α)) =>
        match row with
        | [] => Eq.refl ([] : List (α × α))
        | x :: xs =>
          have h_tail :
              couplingInversePair w (couplingForwardPair w x) ::
                  couplingInverseRow rest (couplingForwardRow rest xs) =
                couplingInversePair w (couplingForwardPair w x) :: xs :=
            congrArg (fun t : List (α × α) => couplingInversePair w (couplingForwardPair w x) :: t)
              (ih xs)
          have h_head :
              couplingInversePair w (couplingForwardPair w x) :: xs = x :: xs :=
            congrArg (fun t : α × α => t :: xs) (couplingPair_left_inverse w x)
          Eq.trans h_tail h_head)
      ws

theorem couplingRow_right_inverse :
    ∀ (ws : List (ElementWeights α)) (row : List (α × α)),
      couplingForwardRow ws (couplingInverseRow ws row) = row :=
  fun ws =>
    List.rec
      (motive := fun qs : List (ElementWeights α) =>
        ∀ row : List (α × α), couplingForwardRow qs (couplingInverseRow qs row) = row)
      (fun row => Eq.refl row)
      (fun (w : ElementWeights α) (rest : List (ElementWeights α))
          (ih : ∀ row : List (α × α), couplingForwardRow rest (couplingInverseRow rest row) = row)
          (row : List (α × α)) =>
        match row with
        | [] => Eq.refl ([] : List (α × α))
        | y :: ys =>
          have h_tail :
              couplingForwardPair w (couplingInversePair w y) ::
                  couplingForwardRow rest (couplingInverseRow rest ys) =
                couplingForwardPair w (couplingInversePair w y) :: ys :=
            congrArg (fun t : List (α × α) => couplingForwardPair w (couplingInversePair w y) :: t)
              (ih ys)
          have h_head :
              couplingForwardPair w (couplingInversePair w y) :: ys = y :: ys :=
            congrArg (fun t : α × α => t :: ys) (couplingPair_right_inverse w y)
          Eq.trans h_tail h_head)
      ws

theorem oftbRow_left_inverse :
    ∀ row : List (α × α), oftbInverseRow (oftbForwardRow row) = row :=
  List.rec
    (motive := fun r : List (α × α) => oftbInverseRow (oftbForwardRow r) = r)
    (Eq.refl ([] : List (α × α)))
    (fun (x : α × α) (xs : List (α × α)) (ih : oftbInverseRow (oftbForwardRow xs) = xs) =>
      have h_tail :
          oftbInversePair (oftbForwardPair x) :: oftbInverseRow (oftbForwardRow xs) =
            oftbInversePair (oftbForwardPair x) :: xs :=
        congrArg (fun t : List (α × α) => oftbInversePair (oftbForwardPair x) :: t) ih
      have h_head : oftbInversePair (oftbForwardPair x) :: xs = x :: xs :=
        congrArg (fun t : α × α => t :: xs) (oftbPair_left_inverse x)
      Eq.trans h_tail h_head)

theorem oftbRow_right_inverse :
    ∀ row : List (α × α), oftbForwardRow (oftbInverseRow row) = row :=
  List.rec
    (motive := fun r : List (α × α) => oftbForwardRow (oftbInverseRow r) = r)
    (Eq.refl ([] : List (α × α)))
    (fun (z : α × α) (zs : List (α × α)) (ih : oftbForwardRow (oftbInverseRow zs) = zs) =>
      have h_tail :
          oftbForwardPair (oftbInversePair z) :: oftbForwardRow (oftbInverseRow zs) =
            oftbForwardPair (oftbInversePair z) :: zs :=
        congrArg (fun t : List (α × α) => oftbForwardPair (oftbInversePair z) :: t) ih
      have h_head : oftbForwardPair (oftbInversePair z) :: zs = z :: zs :=
        congrArg (fun t : α × α => t :: zs) (oftbPair_right_inverse z)
      Eq.trans h_tail h_head)

theorem layerRow_left_inverse (ws : List (ElementWeights α)) (row : List (α × α)) :
    layerInverseRow ws (layerForwardRow ws row) = row :=
  have h1 :
      oftbInverseRow (oftbForwardRow (couplingForwardRow ws row)) =
        couplingForwardRow ws row :=
    oftbRow_left_inverse (couplingForwardRow ws row)
  have h2 :
      couplingInverseRow ws (oftbInverseRow (oftbForwardRow (couplingForwardRow ws row))) =
        couplingInverseRow ws (couplingForwardRow ws row) :=
    congrArg (couplingInverseRow ws) h1
  Eq.trans h2 (couplingRow_left_inverse ws row)

theorem layerRow_right_inverse (ws : List (ElementWeights α)) (row : List (α × α)) :
    layerForwardRow ws (layerInverseRow ws row) = row :=
  have h1 :
      couplingForwardRow ws (couplingInverseRow ws (oftbInverseRow row)) =
        oftbInverseRow row :=
    couplingRow_right_inverse ws (oftbInverseRow row)
  have h2 :
      oftbForwardRow (couplingForwardRow ws (couplingInverseRow ws (oftbInverseRow row))) =
        oftbForwardRow (oftbInverseRow row) :=
    congrArg oftbForwardRow h1
  Eq.trans h2 (oftbRow_right_inverse row)

theorem layerBatch_left_inverse (ws : List (ElementWeights α)) :
    ∀ batch : List (List (α × α)),
      layerInverseBatch ws (layerForwardBatch ws batch) = batch :=
  List.rec
    (motive := fun b : List (List (α × α)) =>
      layerInverseBatch ws (layerForwardBatch ws b) = b)
    (Eq.refl ([] : List (List (α × α))))
    (fun (row : List (α × α)) (rows : List (List (α × α)))
        (ih : layerInverseBatch ws (layerForwardBatch ws rows) = rows) =>
      have h_tail :
          layerInverseRow ws (layerForwardRow ws row) ::
              layerInverseBatch ws (layerForwardBatch ws rows) =
            layerInverseRow ws (layerForwardRow ws row) :: rows :=
        congrArg
          (fun t : List (List (α × α)) => layerInverseRow ws (layerForwardRow ws row) :: t) ih
      have h_head : layerInverseRow ws (layerForwardRow ws row) :: rows = row :: rows :=
        congrArg (fun t : List (α × α) => t :: rows) (layerRow_left_inverse ws row)
      Eq.trans h_tail h_head)

theorem layerBatch_right_inverse (ws : List (ElementWeights α)) :
    ∀ batch : List (List (α × α)),
      layerForwardBatch ws (layerInverseBatch ws batch) = batch :=
  List.rec
    (motive := fun b : List (List (α × α)) =>
      layerForwardBatch ws (layerInverseBatch ws b) = b)
    (Eq.refl ([] : List (List (α × α))))
    (fun (row : List (α × α)) (rows : List (List (α × α)))
        (ih : layerForwardBatch ws (layerInverseBatch ws rows) = rows) =>
      have h_tail :
          layerForwardRow ws (layerInverseRow ws row) ::
              layerForwardBatch ws (layerInverseBatch ws rows) =
            layerForwardRow ws (layerInverseRow ws row) :: rows :=
        congrArg
          (fun t : List (List (α × α)) => layerForwardRow ws (layerInverseRow ws row) :: t) ih
      have h_head : layerForwardRow ws (layerInverseRow ws row) :: rows = row :: rows :=
        congrArg (fun t : List (α × α) => t :: rows) (layerRow_right_inverse ws row)
      Eq.trans h_tail h_head)

theorem stack_left_inverse :
    ∀ (layers : List (List (ElementWeights α))) (batch : List (List (α × α))),
      stackInverse layers (stackForward layers batch) = batch :=
  fun layers =>
    List.rec
      (motive := fun ls : List (List (ElementWeights α)) =>
        ∀ batch : List (List (α × α)), stackInverse ls (stackForward ls batch) = batch)
      (fun batch => Eq.refl batch)
      (fun (ws : List (ElementWeights α)) (rest : List (List (ElementWeights α)))
          (ih : ∀ batch : List (List (α × α)),
            stackInverse rest (stackForward rest batch) = batch)
          (batch : List (List (α × α))) =>
        have h1 :
            layerInverseBatch ws
                (stackInverse rest (stackForward rest (layerForwardBatch ws batch))) =
              layerInverseBatch ws (layerForwardBatch ws batch) :=
          congrArg (layerInverseBatch ws) (ih (layerForwardBatch ws batch))
        Eq.trans h1 (layerBatch_left_inverse ws batch))
      layers

theorem stack_right_inverse :
    ∀ (layers : List (List (ElementWeights α))) (batch : List (List (α × α))),
      stackForward layers (stackInverse layers batch) = batch :=
  fun layers =>
    List.rec
      (motive := fun ls : List (List (ElementWeights α)) =>
        ∀ batch : List (List (α × α)), stackForward ls (stackInverse ls batch) = batch)
      (fun batch => Eq.refl batch)
      (fun (ws : List (ElementWeights α)) (rest : List (List (ElementWeights α)))
          (ih : ∀ batch : List (List (α × α)),
            stackForward rest (stackInverse rest batch) = batch)
          (batch : List (List (α × α))) =>
        have h1 :
            stackForward rest (layerForwardBatch ws (layerInverseBatch ws (stackInverse rest batch))) =
              stackForward rest (stackInverse rest batch) :=
          congrArg (stackForward rest)
            (layerBatch_right_inverse ws (stackInverse rest batch))
        Eq.trans h1 (ih batch))
      layers

theorem stackForward_injective
    (layers : List (List (ElementWeights α)))
    (b1 b2 : List (List (α × α)))
    (h : stackForward layers b1 = stackForward layers b2) : b1 = b2 :=
  have h1 : b1 = stackInverse layers (stackForward layers b1) :=
    Eq.symm (stack_left_inverse layers b1)
  have h2 :
      stackInverse layers (stackForward layers b1) =
        stackInverse layers (stackForward layers b2) :=
    congrArg (stackInverse layers) h
  Eq.trans h1 (Eq.trans h2 (stack_left_inverse layers b2))

theorem stackInverse_injective
    (layers : List (List (ElementWeights α)))
    (b1 b2 : List (List (α × α)))
    (h : stackInverse layers b1 = stackInverse layers b2) : b1 = b2 :=
  have h1 : b1 = stackForward layers (stackInverse layers b1) :=
    Eq.symm (stack_right_inverse layers b1)
  have h2 :
      stackForward layers (stackInverse layers b1) =
        stackForward layers (stackInverse layers b2) :=
    congrArg (stackForward layers) h
  Eq.trans h1 (Eq.trans h2 (stack_right_inverse layers b2))

def listLength {β : Type v} : List β → Nat
  | [] => 0
  | _ :: rest => listLength rest + 1

def appendList {β : Type v} : List β → List β → List β
  | [], ys => ys
  | x :: xs, ys => x :: appendList xs ys

def firsts {β : Type v} {γ : Type v} : List (β × γ) → List β
  | [] => []
  | p :: ps => p.1 :: firsts ps

def seconds {β : Type v} {γ : Type v} : List (β × γ) → List γ
  | [] => []
  | p :: ps => p.2 :: seconds ps

def pairUp {β : Type v} {γ : Type v} : List β → List γ → List (β × γ)
  | [], _ => []
  | _, [] => []
  | a :: as, b :: bs => (a, b) :: pairUp as bs

def takeN {β : Type v} : Nat → List β → List β
  | 0, _ => []
  | _ + 1, [] => []
  | n + 1, x :: xs => x :: takeN n xs

def dropN {β : Type v} : Nat → List β → List β
  | 0, xs => xs
  | _ + 1, [] => []
  | n + 1, _ :: xs => dropN n xs

def rowFlatten {β : Type v} (ps : List (β × β)) : List β :=
  appendList (firsts ps) (seconds ps)

def rowSplit {β : Type v} (half : Nat) (row : List β) : List (β × β) :=
  pairUp (takeN half row) (dropN half row)

theorem takeN_appendList {β : Type v} :
    ∀ (xs ys : List β), takeN (listLength xs) (appendList xs ys) = xs :=
  fun xs =>
    List.rec
      (motive := fun l : List β => ∀ ys : List β, takeN (listLength l) (appendList l ys) = l)
      (fun _ => Eq.refl ([] : List β))
      (fun (x : β) (rest : List β)
          (ih : ∀ ys : List β, takeN (listLength rest) (appendList rest ys) = rest)
          (ys : List β) =>
        congrArg (fun t : List β => x :: t) (ih ys))
      xs

theorem dropN_appendList {β : Type v} :
    ∀ (xs ys : List β), dropN (listLength xs) (appendList xs ys) = ys :=
  fun xs =>
    List.rec
      (motive := fun l : List β => ∀ ys : List β, dropN (listLength l) (appendList l ys) = ys)
      (fun ys => Eq.refl ys)
      (fun (_ : β) (rest : List β)
          (ih : ∀ ys : List β, dropN (listLength rest) (appendList rest ys) = ys)
          (ys : List β) => ih ys)
      xs

theorem pairUp_firsts_seconds {β : Type v} :
    ∀ ps : List (β × β), pairUp (firsts ps) (seconds ps) = ps :=
  List.rec
    (motive := fun l : List (β × β) => pairUp (firsts l) (seconds l) = l)
    (Eq.refl ([] : List (β × β)))
    (fun (p : β × β) (ps : List (β × β)) (ih : pairUp (firsts ps) (seconds ps) = ps) =>
      have h_tail : (p.1, p.2) :: pairUp (firsts ps) (seconds ps) = (p.1, p.2) :: ps :=
        congrArg (fun t : List (β × β) => (p.1, p.2) :: t) ih
      have h_head : (p.1, p.2) :: ps = p :: ps :=
        congrArg (fun t : β × β => t :: ps) (Prod.ext (Eq.refl p.1) (Eq.refl p.2))
      Eq.trans h_tail h_head)

theorem firsts_length {β : Type v} :
    ∀ ps : List (β × β), listLength (firsts ps) = listLength ps :=
  List.rec
    (motive := fun l : List (β × β) => listLength (firsts l) = listLength l)
    (Eq.refl 0)
    (fun (_ : β × β) (ps : List (β × β)) (ih : listLength (firsts ps) = listLength ps) =>
      congrArg (fun n : Nat => n + 1) ih)

theorem rowSplit_rowFlatten {β : Type v} (ps : List (β × β)) :
    rowSplit (listLength (firsts ps)) (rowFlatten ps) = ps :=
  have h_take :
      takeN (listLength (firsts ps)) (appendList (firsts ps) (seconds ps)) = firsts ps :=
    takeN_appendList (firsts ps) (seconds ps)
  have h_drop :
      dropN (listLength (firsts ps)) (appendList (firsts ps) (seconds ps)) = seconds ps :=
    dropN_appendList (firsts ps) (seconds ps)
  have h1 :
      pairUp (takeN (listLength (firsts ps)) (appendList (firsts ps) (seconds ps)))
          (dropN (listLength (firsts ps)) (appendList (firsts ps) (seconds ps))) =
        pairUp (firsts ps)
          (dropN (listLength (firsts ps)) (appendList (firsts ps) (seconds ps))) :=
    congrArg
      (fun t : List β =>
        pairUp t (dropN (listLength (firsts ps)) (appendList (firsts ps) (seconds ps))))
      h_take
  have h2 :
      pairUp (firsts ps)
          (dropN (listLength (firsts ps)) (appendList (firsts ps) (seconds ps))) =
        pairUp (firsts ps) (seconds ps) :=
    congrArg (fun t : List β => pairUp (firsts ps) t) h_drop
  Eq.trans h1 (Eq.trans h2 (pairUp_firsts_seconds ps))

def wordModulus : Nat := 18446744073709551616

theorem wordModulus_pos : 0 < wordModulus :=
  Nat.le_of_ble_eq_true (Eq.refl true)

def wrapAdd (a b : Nat) : Nat := (a + b) % wordModulus

def wrapMul (a b : Nat) : Nat := (a * b) % wordModulus

theorem wrapAdd_lt (a b : Nat) : wrapAdd a b < wordModulus :=
  Nat.mod_lt (a + b) wordModulus_pos

theorem wrapMul_lt (a b : Nat) : wrapMul a b < wordModulus :=
  Nat.mod_lt (a * b) wordModulus_pos

def goldenConstant : Nat := 11400714785074694791

def mixConstant : Nat := 5871781006564002453

def splitMixA : Nat := 13787848793156543929

def splitMixB : Nat := 10723151780598845931

def mixHash (state value : Nat) : Nat :=
  wrapAdd (wrapAdd (wrapMul state goldenConstant) value) mixConstant

theorem mixHash_lt (state value : Nat) : mixHash state value < wordModulus :=
  wrapAdd_lt (wrapAdd (wrapMul state goldenConstant) value) mixConstant

def hashFold (tokens : List Nat) (state : Nat) : Nat :=
  List.foldl mixHash state tokens

theorem hashFold_nil (state : Nat) : hashFold [] state = state :=
  List.foldl_nil

theorem hashFold_cons (token : Nat) (rest : List Nat) (state : Nat) :
    hashFold (token :: rest) state = hashFold rest (mixHash state token) :=
  List.foldl_cons

def hashTokens (tokens : List Nat) : Nat :=
  hashFold tokens (mixHash 0 (listLength tokens))

def computeAnchorHash (tokens : List Nat) (position : Nat) : Nat :=
  hashFold tokens (mixHash position (listLength tokens))

theorem anchorHash_zero_eq_hashTokens (tokens : List Nat) :
    computeAnchorHash tokens 0 = hashTokens tokens :=
  Eq.refl (hashFold tokens (mixHash 0 (listLength tokens)))

theorem hashFold_lt :
    ∀ (tokens : List Nat) (state : Nat),
      state < wordModulus → hashFold tokens state < wordModulus :=
  List.rec
    (motive := fun ts : List Nat =>
      ∀ state : Nat, state < wordModulus → hashFold ts state < wordModulus)
    (fun (state : Nat) (h : state < wordModulus) =>
      Eq.mpr (congrArg (fun t : Nat => t < wordModulus) (hashFold_nil state)) h)
    (fun (token : Nat) (rest : List Nat)
        (ih : ∀ state : Nat, state < wordModulus → hashFold rest state < wordModulus)
        (state : Nat) (_ : state < wordModulus) =>
      Eq.mpr (congrArg (fun t : Nat => t < wordModulus) (hashFold_cons token rest state))
        (ih (mixHash state token) (mixHash_lt state token)))

theorem hashTokens_lt (tokens : List Nat) : hashTokens tokens < wordModulus :=
  hashFold_lt tokens (mixHash 0 (listLength tokens)) (mixHash_lt 0 (listLength tokens))

theorem anchorHash_lt (tokens : List Nat) (position : Nat) :
    computeAnchorHash tokens position < wordModulus :=
  hashFold_lt tokens (mixHash position (listLength tokens))
    (mixHash_lt position (listLength tokens))

theorem hashFold_appendList :
    ∀ (xs ys : List Nat) (state : Nat),
      hashFold (appendList xs ys) state = hashFold ys (hashFold xs state) :=
  fun xs =>
    List.rec
      (motive := fun l : List Nat =>
        ∀ (ys : List Nat) (state : Nat),
          hashFold (appendList l ys) state = hashFold ys (hashFold l state))
      (fun (ys : List Nat) (state : Nat) =>
        congrArg (fun t : Nat => hashFold ys t) (Eq.symm (hashFold_nil state)))
      (fun (token : Nat) (rest : List Nat)
          (ih : ∀ (ys : List Nat) (state : Nat),
            hashFold (appendList rest ys) state = hashFold ys (hashFold rest state))
          (ys : List Nat) (state : Nat) =>
        Eq.trans (hashFold_cons token (appendList rest ys) state)
          (Eq.trans (ih ys (mixHash state token))
            (congrArg (fun t : Nat => hashFold ys t)
              (Eq.symm (hashFold_cons token rest state)))))
      xs

def bucketWidth : Nat := 6

def bucketCount : Nat := 1 <<< bucketWidth

def bucketMask : Nat := bucketCount - 1

def splitMixStep30 (h : Nat) : Nat := wrapMul (h ^^^ (h >>> 30)) splitMixA

def splitMixStep27 (h : Nat) : Nat := wrapMul (h ^^^ (h >>> 27)) splitMixB

def splitMixStep31 (h : Nat) : Nat := h ^^^ (h >>> 31)

def bucketIndex (position : Nat) : Nat :=
  splitMixStep31 (splitMixStep27 (splitMixStep30 (wrapMul position goldenConstant))) &&& bucketMask

theorem bucketCount_pos : 0 < bucketCount :=
  Nat.le_of_ble_eq_true (Eq.refl true)

theorem bucketMask_lt : bucketMask < bucketCount :=
  Nat.le_of_ble_eq_true (Eq.refl true)

theorem bucketIndex_lt (position : Nat) : bucketIndex position < bucketCount :=
  Nat.lt_of_le_of_lt Nat.and_le_right bucketMask_lt

def segmentFullHash (position scoreBits anchorHash signature : Nat) : Nat :=
  mixHash (mixHash (mixHash (mixHash 0 position) scoreBits) anchorHash) signature

theorem segmentFullHash_lt (position scoreBits anchorHash signature : Nat) :
    segmentFullHash position scoreBits anchorHash signature < wordModulus :=
  mixHash_lt (mixHash (mixHash (mixHash 0 position) scoreBits) anchorHash) signature

theorem appendList_length {β : Type v} :
    ∀ xs ys : List β, listLength (appendList xs ys) = listLength xs + listLength ys :=
  fun xs =>
    List.rec
      (motive := fun l : List β =>
        ∀ ys : List β, listLength (appendList l ys) = listLength l + listLength ys)
      (fun ys : List β => Eq.symm (Nat.zero_add (listLength ys)))
      (fun (_ : β) (rest : List β)
          (ih : ∀ ys : List β, listLength (appendList rest ys) = listLength rest + listLength ys)
          (ys : List β) =>
        Eq.trans (congrArg (fun n : Nat => n + 1) (ih ys))
          (Eq.symm (Nat.succ_add (listLength rest) (listLength ys))))
      xs

def tokenToLEBytes (token : Nat) : List Nat :=
  [token % 256, (token >>> 8) % 256, (token >>> 16) % 256, (token >>> 24) % 256]

def encodeNgramLE : List Nat → List Nat
  | [] => []
  | token :: rest => appendList (tokenToLEBytes token) (encodeNgramLE rest)

theorem byteMod_lt (value : Nat) : value % 256 < 256 :=
  Nat.mod_lt value (Nat.le_of_ble_eq_true (Eq.refl true))

theorem tokenToLEBytes_length (token : Nat) : listLength (tokenToLEBytes token) = 4 :=
  Eq.refl 4

theorem encodeNgramLE_length :
    ∀ tokens : List Nat, listLength (encodeNgramLE tokens) = listLength tokens * 4 :=
  List.rec
    (motive := fun ts : List Nat => listLength (encodeNgramLE ts) = listLength ts * 4)
    (Eq.refl 0)
    (fun (token : Nat) (rest : List Nat)
        (ih : listLength (encodeNgramLE rest) = listLength rest * 4) =>
      Eq.trans (appendList_length (tokenToLEBytes token) (encodeNgramLE rest))
        (Eq.trans (congrArg (fun n : Nat => 4 + n) ih)
          (Eq.trans (Nat.add_comm 4 (listLength rest * 4))
            (Eq.symm (Nat.succ_mul (listLength rest) 4)))))

def usizeMax : Nat := 18446744073709551615

def satAddUsize (a b : Nat) : Nat :=
  match Nat.ble (a + b) usizeMax with
  | true => a + b
  | false => usizeMax

def satSubUsize (a b : Nat) : Nat :=
  match Nat.blt a b with
  | true => 0
  | false => a - b

theorem satAddUsize_le (a b : Nat) : satAddUsize a b ≤ usizeMax :=
  Bool.rec
    (motive := fun t : Bool =>
      Nat.ble (a + b) usizeMax = t →
        (match t with | true => a + b | false => usizeMax) ≤ usizeMax)
    (fun _ => Nat.le_refl usizeMax)
    (fun h => Nat.le_of_ble_eq_true h)
    (Nat.ble (a + b) usizeMax)
    (Eq.refl (Nat.ble (a + b) usizeMax))

theorem satSubUsize_le (a b : Nat) : satSubUsize a b ≤ a :=
  Bool.rec
    (motive := fun t : Bool =>
      Nat.blt a b = t → (match t with | true => 0 | false => a - b) ≤ a)
    (fun _ => Nat.sub_le a b)
    (fun _ => Nat.zero_le a)
    (Nat.blt a b)
    (Eq.refl (Nat.blt a b))

def vpuWordIndex (wordsPerRow row col : Nat) : Nat :=
  row * wordsPerRow + (col >>> 6)

def vpuTransposedWordIndex (wordsPerRow row col : Nat) : Nat :=
  col * wordsPerRow + (row >>> 6)

def vpuBitIndex (index : Nat) : Nat := index &&& 63

theorem vpuBitIndex_lt (index : Nat) : vpuBitIndex index < 64 :=
  Nat.lt_of_le_of_lt Nat.and_le_right (Nat.le_of_ble_eq_true (Eq.refl true))

theorem rowMajorIndex_lt (rows width row offset : Nat)
    (hrow : row < rows) (hoffset : offset < width) :
    row * width + offset < rows * width :=
  Nat.lt_of_lt_of_le
    (Nat.lt_of_lt_of_le (Nat.add_lt_add_left hoffset (row * width))
      (Nat.le_of_eq (Eq.symm (Nat.succ_mul row width))))
    (Nat.mul_le_mul_right width (Nat.succ_le_of_lt hrow))

theorem vpuWordIndex_lt (wordsPerRow rows row col : Nat)
    (hrow : row < rows) (hcol : col >>> 6 < wordsPerRow) :
    vpuWordIndex wordsPerRow row col < rows * wordsPerRow :=
  rowMajorIndex_lt rows wordsPerRow row (col >>> 6) hrow hcol

theorem vpuTransposedWordIndex_lt (wordsPerRow cols row col : Nat)
    (hcol : col < cols) (hrow : row >>> 6 < wordsPerRow) :
    vpuTransposedWordIndex wordsPerRow row col < cols * wordsPerRow :=
  rowMajorIndex_lt cols wordsPerRow col (row >>> 6) hcol hrow

def peerMatrixIndex (deviceCount src dst : Nat) : Nat := src * deviceCount + dst

theorem peerMatrixIndex_lt (deviceCount src dst : Nat)
    (hsrc : src < deviceCount) (hdst : dst < deviceCount) :
    peerMatrixIndex deviceCount src dst < deviceCount * deviceCount :=
  rowMajorIndex_lt deviceCount deviceCount src dst hsrc hdst

def highNibble (byte : Nat) : Nat := (byte >>> 4) &&& 15

def lowNibble (byte : Nat) : Nat := byte &&& 15

theorem highNibble_lt (byte : Nat) : highNibble byte < 16 :=
  Nat.lt_of_le_of_lt Nat.and_le_right (Nat.le_of_ble_eq_true (Eq.refl true))

theorem lowNibble_lt (byte : Nat) : lowNibble byte < 16 :=
  Nat.lt_of_le_of_lt Nat.and_le_right (Nat.le_of_ble_eq_true (Eq.refl true))

def hexTable : List Nat :=
  [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 97, 98, 99, 100, 101, 102]

def natListIndex : List Nat → Nat → Nat → Nat
  | [], _, fallback => fallback
  | value :: _, 0, _ => value
  | _ :: rest, n + 1, fallback => natListIndex rest n fallback

def memNat : Nat → List Nat → Bool
  | _, [] => false
  | value, entry :: rest =>
      match Nat.beq value entry with
      | true => true
      | false => memNat value rest

def allMemNat : List Nat → List Nat → Bool
  | [], _ => true
  | value :: rest, alphabet =>
      match memNat value alphabet with
      | true => allMemNat rest alphabet
      | false => false

theorem natListIndex_mem :
    ∀ (alphabet : List Nat) (index fallback : Nat),
      index < listLength alphabet → memNat (natListIndex alphabet index fallback) alphabet = true :=
  List.rec
    (motive := fun xs : List Nat =>
      ∀ index fallback : Nat,
        index < listLength xs → memNat (natListIndex xs index fallback) xs = true)
    (fun (index : Nat) (_ : Nat) (h : index < 0) => absurd h (Nat.not_lt_zero index))
    (fun (entry : Nat) (rest : List Nat)
        (ih : ∀ index fallback : Nat,
          index < listLength rest → memNat (natListIndex rest index fallback) rest = true)
        (index : Nat) =>
      Nat.rec
        (motive := fun n : Nat =>
          ∀ fallback : Nat,
            n < listLength rest + 1 →
              memNat (natListIndex (entry :: rest) n fallback) (entry :: rest) = true)
        (fun (_ : Nat) (_ : 0 < listLength rest + 1) =>
          congrArg
            (fun t : Bool => match t with | true => true | false => memNat entry rest)
            (Nat.beq_refl entry))
        (fun (n : Nat)
            (_ : ∀ fallback : Nat,
              n < listLength rest + 1 →
                memNat (natListIndex (entry :: rest) n fallback) (entry :: rest) = true)
            (fallback : Nat) (hn : n + 1 < listLength rest + 1) =>
          Bool.rec
            (motive := fun t : Bool =>
              (match t with
                | true => true
                | false => memNat (natListIndex rest n fallback) rest) = true)
            (ih n fallback (Nat.lt_of_succ_lt_succ hn))
            (Eq.refl true)
            (Nat.beq (natListIndex rest n fallback) entry))
        index)

def nibbleChar (nibble : Nat) : Nat := natListIndex hexTable nibble 48

theorem hexTable_length : listLength hexTable = 16 := Eq.refl 16

theorem nibbleChar_mem (nibble : Nat) (h : nibble < 16) :
    memNat (nibbleChar nibble) hexTable = true :=
  natListIndex_mem hexTable nibble 48 h

def bytesToHexLower : List Nat → List Nat
  | [] => []
  | byte :: rest =>
      nibbleChar (highNibble byte) :: nibbleChar (lowNibble byte) :: bytesToHexLower rest

theorem bytesToHexLower_length :
    ∀ bytes : List Nat,
      listLength (bytesToHexLower bytes) = listLength bytes + listLength bytes :=
  List.rec
    (motive := fun bs : List Nat =>
      listLength (bytesToHexLower bs) = listLength bs + listLength bs)
    (Eq.refl 0)
    (fun (_ : Nat) (rest : List Nat)
        (ih : listLength (bytesToHexLower rest) = listLength rest + listLength rest) =>
      Eq.trans
        (congrArg (fun n : Nat => n + 1 + 1) ih)
        (Eq.trans
          (congrArg (fun n : Nat => n + 1) (Eq.symm (Nat.succ_add (listLength rest) (listLength rest))))
          (Eq.symm (Nat.add_succ (listLength rest + 1) (listLength rest)))))

def stableMonotonic (previous observed : Nat) : Nat :=
  match Nat.blt observed previous with
  | true => previous
  | false => observed

theorem stableMonotonic_ge_previous (previous observed : Nat) :
    previous ≤ stableMonotonic previous observed :=
  Bool.rec
    (motive := fun t : Bool =>
      Nat.blt observed previous = t →
        previous ≤ (match t with | true => previous | false => observed))
    (fun h =>
      Nat.le_of_not_lt (fun hlt : observed < previous =>
        Bool.noConfusion (Eq.trans (Eq.symm (Nat.ble_eq_true_of_le hlt)) h)))
    (fun _ => Nat.le_refl previous)
    (Nat.blt observed previous)
    (Eq.refl (Nat.blt observed previous))

def lengthBytesLE (value : Nat) : List Nat :=
  [value % 256, (value >>> 8) % 256, (value >>> 16) % 256, (value >>> 24) % 256,
   (value >>> 32) % 256, (value >>> 40) % 256, (value >>> 48) % 256, (value >>> 56) % 256]

def frameSlice (payload : List Nat) : List Nat :=
  appendList (lengthBytesLE (listLength payload)) payload

theorem lengthBytesLE_length (value : Nat) : listLength (lengthBytesLE value) = 8 :=
  Eq.refl 8

theorem frameSlice_payload (payload : List Nat) : dropN 8 (frameSlice payload) = payload :=
  dropN_appendList (lengthBytesLE (listLength payload)) payload

theorem frameSlice_prefix (payload : List Nat) :
    takeN 8 (frameSlice payload) = lengthBytesLE (listLength payload) :=
  takeN_appendList (lengthBytesLE (listLength payload)) payload

def edgeKeyBytes (source target : List Nat) : List Nat :=
  appendList (frameSlice source) (frameSlice target)

theorem edgeKeyBytes_source (source target : List Nat) :
    takeN 8 (edgeKeyBytes source target) = lengthBytesLE (listLength source) :=
  takeN_appendList (lengthBytesLE (listLength source))
    (appendList source (frameSlice target))

inductive SymmetryGroup where
  | identity : SymmetryGroup
  | reflection : SymmetryGroup
  | rotation90 : SymmetryGroup
  | rotation180 : SymmetryGroup
  | rotation270 : SymmetryGroup
  | translation : SymmetryGroup
  | customRotation : SymmetryGroup

def symmetryOrder : SymmetryGroup → Nat
  | SymmetryGroup.identity => 1
  | SymmetryGroup.reflection => 2
  | SymmetryGroup.rotation90 => 4
  | SymmetryGroup.rotation180 => 2
  | SymmetryGroup.rotation270 => 4
  | SymmetryGroup.translation => 1
  | SymmetryGroup.customRotation => 2

theorem symmetryOrder_pos (group : SymmetryGroup) : 0 < symmetryOrder group :=
  SymmetryGroup.rec
    (motive := fun g : SymmetryGroup => 0 < symmetryOrder g)
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    group

theorem symmetryOrder_le (group : SymmetryGroup) : symmetryOrder group ≤ 4 :=
  SymmetryGroup.rec
    (motive := fun g : SymmetryGroup => symmetryOrder g ≤ 4)
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    group

def natListEq : List Nat → List Nat → Bool
  | [], [] => true
  | [], _ :: _ => false
  | _ :: _, [] => false
  | a :: as, b :: bs =>
      match Nat.beq a b with
      | true => natListEq as bs
      | false => false

theorem natListEq_refl : ∀ xs : List Nat, natListEq xs xs = true :=
  List.rec
    (motive := fun l : List Nat => natListEq l l = true)
    (Eq.refl true)
    (fun (a : Nat) (rest : List Nat) (ih : natListEq rest rest = true) =>
      Eq.trans
        (congrArg
          (fun t : Bool => match t with | true => natListEq rest rest | false => false)
          (Nat.beq_refl a))
        ih)

def wordsMatchStem (stemOf : List Nat → List Nat) (a b : List Nat) : Bool :=
  match natListEq a b with
  | true => true
  | false =>
      match natListEq (stemOf a) (stemOf b) with
      | true => true
      | false => false

theorem wordsMatchStem_refl (stemOf : List Nat → List Nat) (word : List Nat) :
    wordsMatchStem stemOf word word = true :=
  congrArg
    (fun t : Bool =>
      match t with
      | true => true
      | false =>
          match natListEq (stemOf word) (stemOf word) with
          | true => true
          | false => false)
    (natListEq_refl word)

def historyPush {β : Type v} (history : List β) (entry : β) : List β :=
  appendList history (entry :: [])

theorem historyPush_length {β : Type v} (history : List β) (entry : β) :
    listLength (historyPush history entry) = listLength history + 1 :=
  appendList_length history (entry :: [])


def bitAt : List Bool → Nat → Bool
  | [], _ => false
  | b :: _, 0 => b
  | _ :: rest, n + 1 => bitAt rest n

def setBitAt : List Bool → Nat → List Bool
  | [], _ => []
  | _ :: rest, 0 => true :: rest
  | b :: rest, n + 1 => b :: setBitAt rest n

def clearBitAt : List Bool → Nat → List Bool
  | [], _ => []
  | _ :: rest, 0 => false :: rest
  | b :: rest, n + 1 => b :: clearBitAt rest n

def popCountBits : List Bool → Nat
  | [] => 0
  | true :: rest => popCountBits rest + 1
  | false :: rest => popCountBits rest

theorem bitAt_setBitAt :
    ∀ (bits : List Bool) (index : Nat),
      index < listLength bits → bitAt (setBitAt bits index) index = true :=
  List.rec
    (motive := fun bs : List Bool =>
      ∀ index : Nat, index < listLength bs → bitAt (setBitAt bs index) index = true)
    (fun (index : Nat) (h : index < 0) => absurd h (Nat.not_lt_zero index))
    (fun (b : Bool) (rest : List Bool)
        (ih : ∀ index : Nat, index < listLength rest → bitAt (setBitAt rest index) index = true)
        (index : Nat) =>
      Nat.rec
        (motive := fun n : Nat =>
          n < listLength rest + 1 → bitAt (setBitAt (b :: rest) n) n = true)
        (fun _ => Eq.refl true)
        (fun (n : Nat) (_ : n < listLength rest + 1 → bitAt (setBitAt (b :: rest) n) n = true)
            (hn : n + 1 < listLength rest + 1) =>
          ih n (Nat.lt_of_succ_lt_succ hn))
        index)

theorem bitAt_clearBitAt :
    ∀ (bits : List Bool) (index : Nat),
      index < listLength bits → bitAt (clearBitAt bits index) index = false :=
  List.rec
    (motive := fun bs : List Bool =>
      ∀ index : Nat, index < listLength bs → bitAt (clearBitAt bs index) index = false)
    (fun (index : Nat) (h : index < 0) => absurd h (Nat.not_lt_zero index))
    (fun (b : Bool) (rest : List Bool)
        (ih : ∀ index : Nat, index < listLength rest → bitAt (clearBitAt rest index) index = false)
        (index : Nat) =>
      Nat.rec
        (motive := fun n : Nat =>
          n < listLength rest + 1 → bitAt (clearBitAt (b :: rest) n) n = false)
        (fun _ => Eq.refl false)
        (fun (n : Nat) (_ : n < listLength rest + 1 → bitAt (clearBitAt (b :: rest) n) n = false)
            (hn : n + 1 < listLength rest + 1) =>
          ih n (Nat.lt_of_succ_lt_succ hn))
        index)

theorem popCountBits_le :
    ∀ bits : List Bool, popCountBits bits ≤ listLength bits :=
  List.rec
    (motive := fun bs : List Bool => popCountBits bs ≤ listLength bs)
    (Nat.le_refl 0)
    (fun (b : Bool) (rest : List Bool) (ih : popCountBits rest ≤ listLength rest) =>
      Bool.rec
        (motive := fun t : Bool => popCountBits (t :: rest) ≤ listLength rest + 1)
        (Nat.le_trans ih (Nat.le_succ (listLength rest)))
        (Nat.succ_le_succ ih)
        b)

def matrixRowAt : List (List Bool) → Nat → List Bool
  | [], _ => []
  | row :: _, 0 => row
  | _ :: rest, n + 1 => matrixRowAt rest n

def matrixSetBit : List (List Bool) → Nat → Nat → List (List Bool)
  | [], _, _ => []
  | row :: rest, 0, col => setBitAt row col :: rest
  | row :: rest, n + 1, col => row :: matrixSetBit rest n col

def matrixGetBit (matrix : List (List Bool)) (row col : Nat) : Bool :=
  bitAt (matrixRowAt matrix row) col

theorem matrixGetBit_setBit :
    ∀ (matrix : List (List Bool)) (row col : Nat),
      row < listLength matrix → col < listLength (matrixRowAt matrix row) →
        matrixGetBit (matrixSetBit matrix row col) row col = true :=
  List.rec
    (motive := fun m : List (List Bool) =>
      ∀ row col : Nat,
        row < listLength m → col < listLength (matrixRowAt m row) →
          matrixGetBit (matrixSetBit m row col) row col = true)
    (fun (row : Nat) (_ : Nat) (h : row < 0) => absurd h (Nat.not_lt_zero row))
    (fun (head : List Bool) (rest : List (List Bool))
        (ih : ∀ row col : Nat,
          row < listLength rest → col < listLength (matrixRowAt rest row) →
            matrixGetBit (matrixSetBit rest row col) row col = true)
        (row : Nat) =>
      Nat.rec
        (motive := fun n : Nat =>
          ∀ col : Nat,
            n < listLength rest + 1 → col < listLength (matrixRowAt (head :: rest) n) →
              matrixGetBit (matrixSetBit (head :: rest) n col) n col = true)
        (fun (col : Nat) (_ : 0 < listLength rest + 1) (hcol : col < listLength head) =>
          bitAt_setBitAt head col hcol)
        (fun (n : Nat)
            (_ : ∀ col : Nat,
              n < listLength rest + 1 → col < listLength (matrixRowAt (head :: rest) n) →
                matrixGetBit (matrixSetBit (head :: rest) n col) n col = true)
            (col : Nat) (hrow : n + 1 < listLength rest + 1)
            (hcol : col < listLength (matrixRowAt rest n)) =>
          ih n col (Nat.lt_of_succ_lt_succ hrow) hcol)
        row)

def mirroredSet (matrix transposed : List (List Bool)) (row col : Nat) :
    List (List Bool) × List (List Bool) :=
  (matrixSetBit matrix row col, matrixSetBit transposed col row)

theorem mirroredSet_agrees
    (matrix transposed : List (List Bool)) (row col : Nat)
    (hrow : row < listLength matrix)
    (hcol : col < listLength (matrixRowAt matrix row))
    (hcolT : col < listLength transposed)
    (hrowT : row < listLength (matrixRowAt transposed col)) :
    matrixGetBit (mirroredSet matrix transposed row col).1 row col =
      matrixGetBit (mirroredSet matrix transposed row col).2 col row :=
  Eq.trans (matrixGetBit_setBit matrix row col hrow hcol)
    (Eq.symm (matrixGetBit_setBit transposed col row hcolT hrowT))

def lookupAssoc {β : Type v} : List (List Nat × β) → List Nat → Option β
  | [], _ => none
  | entry :: rest, key =>
      match natListEq entry.1 key with
      | true => some entry.2
      | false => lookupAssoc rest key

def removeAssoc {β : Type v} : List (List Nat × β) → List Nat → List (List Nat × β)
  | [], _ => []
  | entry :: rest, key =>
      match natListEq entry.1 key with
      | true => removeAssoc rest key
      | false => entry :: removeAssoc rest key

def insertAssoc {β : Type v} (table : List (List Nat × β)) (key : List Nat) (value : β) :
    List (List Nat × β) :=
  (key, value) :: removeAssoc table key

theorem lookupAssoc_insertAssoc {β : Type v}
    (table : List (List Nat × β)) (key : List Nat) (value : β) :
    lookupAssoc (insertAssoc table key value) key = some value :=
  congrArg
    (fun t : Bool =>
      match t with
      | true => some value
      | false => lookupAssoc (removeAssoc table key) key)
    (natListEq_refl key)

theorem lookupAssoc_removeAssoc {β : Type v} :
    ∀ (table : List (List Nat × β)) (key : List Nat),
      lookupAssoc (removeAssoc table key) key = none :=
  fun table =>
    List.rec
      (motive := fun t : List (List Nat × β) =>
        ∀ key : List Nat, lookupAssoc (removeAssoc t key) key = none)
      (fun _ => Eq.refl none)
      (fun (entry : List Nat × β) (rest : List (List Nat × β))
          (ih : ∀ key : List Nat, lookupAssoc (removeAssoc rest key) key = none)
          (key : List Nat) =>
        Bool.rec
          (motive := fun t : Bool =>
            natListEq entry.1 key = t →
              lookupAssoc
                  (match t with
                    | true => removeAssoc rest key
                    | false => entry :: removeAssoc rest key)
                  key =
                none)
          (fun h =>
            Eq.trans
              (congrArg
                (fun t : Bool =>
                  match t with
                  | true => some entry.2
                  | false => lookupAssoc (removeAssoc rest key) key)
                h)
              (ih key))
          (fun _ => ih key)
          (natListEq entry.1 key)
          (Eq.refl (natListEq entry.1 key)))
      table

theorem removeAssoc_length {β : Type v} :
    ∀ (table : List (List Nat × β)) (key : List Nat),
      listLength (removeAssoc table key) ≤ listLength table :=
  fun table =>
    List.rec
      (motive := fun t : List (List Nat × β) =>
        ∀ key : List Nat, listLength (removeAssoc t key) ≤ listLength t)
      (fun _ => Nat.le_refl 0)
      (fun (entry : List Nat × β) (rest : List (List Nat × β))
          (ih : ∀ key : List Nat, listLength (removeAssoc rest key) ≤ listLength rest)
          (key : List Nat) =>
        Bool.rec
          (motive := fun t : Bool =>
            listLength
                (match t with
                  | true => removeAssoc rest key
                  | false => entry :: removeAssoc rest key) ≤
              listLength rest + 1)
          (Nat.succ_le_succ (ih key))
          (Nat.le_trans (ih key) (Nat.le_succ (listLength rest)))
          (natListEq entry.1 key))
      table

def retainAbove : Nat → List Nat → List Nat
  | _, [] => []
  | threshold, score :: rest =>
      match Nat.blt threshold score with
      | true => score :: retainAbove threshold rest
      | false => retainAbove threshold rest

theorem retainAbove_length :
    ∀ (threshold : Nat) (scores : List Nat),
      listLength (retainAbove threshold scores) ≤ listLength scores :=
  fun threshold =>
    List.rec
      (motive := fun ss : List Nat =>
        listLength (retainAbove threshold ss) ≤ listLength ss)
      (Nat.le_refl 0)
      (fun (score : Nat) (rest : List Nat)
          (ih : listLength (retainAbove threshold rest) ≤ listLength rest) =>
        Bool.rec
          (motive := fun t : Bool =>
            listLength
                (match t with
                  | true => score :: retainAbove threshold rest
                  | false => retainAbove threshold rest) ≤
              listLength rest + 1)
          (Nat.le_trans ih (Nat.le_succ (listLength rest)))
          (Nat.succ_le_succ ih)
          (Nat.blt threshold score))

def allAbove : Nat → List Nat → Bool
  | _, [] => true
  | threshold, score :: rest =>
      match Nat.blt threshold score with
      | true => allAbove threshold rest
      | false => false

theorem retainAbove_allAbove :
    ∀ (threshold : Nat) (scores : List Nat),
      allAbove threshold (retainAbove threshold scores) = true :=
  fun threshold =>
    List.rec
      (motive := fun ss : List Nat => allAbove threshold (retainAbove threshold ss) = true)
      (Eq.refl true)
      (fun (score : Nat) (rest : List Nat)
          (ih : allAbove threshold (retainAbove threshold rest) = true) =>
        Bool.rec
          (motive := fun t : Bool =>
            Nat.blt threshold score = t →
              allAbove threshold
                  (match t with
                    | true => score :: retainAbove threshold rest
                    | false => retainAbove threshold rest) =
                true)
          (fun _ => ih)
          (fun h =>
            Eq.trans
              (congrArg
                (fun t : Bool =>
                  match t with
                  | true => allAbove threshold (retainAbove threshold rest)
                  | false => false)
                h)
              ih)
          (Nat.blt threshold score)
          (Eq.refl (Nat.blt threshold score)))

def updateBestEnergy (best current : Nat) : Nat :=
  match Nat.blt current best with
  | true => current
  | false => best

theorem updateBestEnergy_le (best current : Nat) : updateBestEnergy best current ≤ best :=
  Bool.rec
    (motive := fun t : Bool =>
      Nat.blt current best = t → (match t with | true => current | false => best) ≤ best)
    (fun _ => Nat.le_refl best)
    (fun h => Nat.le_of_lt (Nat.le_of_ble_eq_true h))
    (Nat.blt current best)
    (Eq.refl (Nat.blt current best))

def runBestEnergy : List Nat → Nat → Nat
  | [], best => best
  | current :: rest, best => runBestEnergy rest (updateBestEnergy best current)

theorem runBestEnergy_le :
    ∀ (energies : List Nat) (best : Nat), runBestEnergy energies best ≤ best :=
  List.rec
    (motive := fun es : List Nat => ∀ best : Nat, runBestEnergy es best ≤ best)
    (fun best => Nat.le_refl best)
    (fun (current : Nat) (rest : List Nat)
        (ih : ∀ best : Nat, runBestEnergy rest best ≤ best) (best : Nat) =>
      Nat.le_trans (ih (updateBestEnergy best current)) (updateBestEnergy_le best current))

def rankedLess (left right : Nat) : Bool := Nat.blt right left

theorem takeN_length_le {β : Type v} :
    ∀ (count : Nat) (items : List β), listLength (takeN count items) ≤ count :=
  Nat.rec
    (motive := fun n : Nat => ∀ items : List β, listLength (takeN n items) ≤ n)
    (fun _ => Nat.le_refl 0)
    (fun (n : Nat) (ih : ∀ items : List β, listLength (takeN n items) ≤ n) (items : List β) =>
      List.rec
        (motive := fun l : List β => listLength (takeN (n + 1) l) ≤ n + 1)
        (Nat.zero_le (n + 1))
        (fun (_ : β) (rest : List β) (_ : listLength (takeN (n + 1) rest) ≤ n + 1) =>
          Nat.succ_le_succ (ih rest))
        items)

def edgeKeyEq (leftSource leftTarget rightSource rightTarget : List Nat) : Bool :=
  match natListEq leftSource rightSource with
  | true => natListEq leftTarget rightTarget
  | false => false

theorem edgeKeyEq_refl (source target : List Nat) :
    edgeKeyEq source target source target = true :=
  Eq.trans
    (congrArg
      (fun t : Bool =>
        match t with
        | true => natListEq target target
        | false => false)
      (natListEq_refl source))
    (natListEq_refl target)

def findEdge : List (List Nat × List Nat) → List Nat → List Nat → Bool
  | [], _, _ => false
  | edge :: rest, source, target =>
      match edgeKeyEq edge.1 edge.2 source target with
      | true => true
      | false => findEdge rest source target

def addEdge (edges : List (List Nat × List Nat)) (source target : List Nat) :
    List (List Nat × List Nat) :=
  (source, target) :: edges

theorem findEdge_addEdge (edges : List (List Nat × List Nat)) (source target : List Nat) :
    findEdge (addEdge edges source target) source target = true :=
  congrArg
    (fun t : Bool =>
      match t with
      | true => true
      | false => findEdge edges source target)
    (edgeKeyEq_refl source target)

theorem addEdge_length (edges : List (List Nat × List Nat)) (source target : List Nat) :
    listLength (addEdge edges source target) = listLength edges + 1 :=
  Eq.refl (listLength edges + 1)

def drainQueue {β : Type v} : List β → List β × Nat
  | [] => ([], 0)
  | _ :: rest => ((drainQueue rest).1, (drainQueue rest).2 + 1)

theorem drainQueue_conserves {β : Type v} :
    ∀ queue : List β, (drainQueue queue).2 = listLength queue :=
  List.rec
    (motive := fun q : List β => (drainQueue q).2 = listLength q)
    (Eq.refl 0)
    (fun (_ : β) (rest : List β) (ih : (drainQueue rest).2 = listLength rest) =>
      congrArg (fun n : Nat => n + 1) ih)

theorem drainQueue_empties {β : Type v} :
    ∀ queue : List β, listLength (drainQueue queue).1 = 0 :=
  List.rec
    (motive := fun q : List β => listLength (drainQueue q).1 = 0)
    (Eq.refl 0)
    (fun (_ : β) (rest : List β) (ih : listLength (drainQueue rest).1 = 0) => ih)

def nextDeviceIndex (current deviceCount : Nat) : Nat := (current + 1) % deviceCount

theorem nextDeviceIndex_lt (current deviceCount : Nat) (h : 0 < deviceCount) :
    nextDeviceIndex current deviceCount < deviceCount :=
  Nat.mod_lt (current + 1) h

def probeIndex (hash step capacity : Nat) : Nat := (hash + step) % capacity

theorem probeIndex_lt (hash step capacity : Nat) (h : 0 < capacity) :
    probeIndex hash step capacity < capacity :=
  Nat.mod_lt (hash + step) h

inductive MemoryBlockState where
  | free : MemoryBlockState
  | allocated : MemoryBlockState
  | entangled : MemoryBlockState
  | migrating : MemoryBlockState

def memoryBlockStateCode : MemoryBlockState → Nat
  | MemoryBlockState.free => 0
  | MemoryBlockState.allocated => 1
  | MemoryBlockState.entangled => 2
  | MemoryBlockState.migrating => 3

theorem memoryBlockStateCode_lt (state : MemoryBlockState) : memoryBlockStateCode state < 4 :=
  MemoryBlockState.rec
    (motive := fun s : MemoryBlockState => memoryBlockStateCode s < 4)
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    (Nat.le_of_ble_eq_true (Eq.refl true))
    state

def allocateBlock : MemoryBlockState → MemoryBlockState
  | MemoryBlockState.free => MemoryBlockState.allocated
  | MemoryBlockState.allocated => MemoryBlockState.allocated
  | MemoryBlockState.entangled => MemoryBlockState.entangled
  | MemoryBlockState.migrating => MemoryBlockState.migrating

def freeBlock : MemoryBlockState → MemoryBlockState
  | MemoryBlockState.free => MemoryBlockState.free
  | MemoryBlockState.allocated => MemoryBlockState.free
  | MemoryBlockState.entangled => MemoryBlockState.free
  | MemoryBlockState.migrating => MemoryBlockState.migrating

theorem allocate_free_roundtrip :
    freeBlock (allocateBlock MemoryBlockState.free) = MemoryBlockState.free :=
  Eq.refl MemoryBlockState.free

theorem allocateBlock_idempotent (state : MemoryBlockState) :
    allocateBlock (allocateBlock state) = allocateBlock state :=
  MemoryBlockState.rec
    (motive := fun s : MemoryBlockState => allocateBlock (allocateBlock s) = allocateBlock s)
    (Eq.refl MemoryBlockState.allocated)
    (Eq.refl MemoryBlockState.allocated)
    (Eq.refl MemoryBlockState.entangled)
    (Eq.refl MemoryBlockState.migrating)
    state


theorem bytesToHexLower_alphabet :
    ∀ bytes : List Nat, allMemNat (bytesToHexLower bytes) hexTable = true :=
  List.rec
    (motive := fun bs : List Nat => allMemNat (bytesToHexLower bs) hexTable = true)
    (Eq.refl true)
    (fun (byte : Nat) (rest : List Nat)
        (ih : allMemNat (bytesToHexLower rest) hexTable = true) =>
      Eq.trans
        (congrArg
          (fun t : Bool =>
            match t with
            | true =>
                match memNat (nibbleChar (lowNibble byte)) hexTable with
                | true => allMemNat (bytesToHexLower rest) hexTable
                | false => false
            | false => false)
          (nibbleChar_mem (highNibble byte) (highNibble_lt byte)))
        (Eq.trans
          (congrArg
            (fun t : Bool =>
              match t with
              | true => allMemNat (bytesToHexLower rest) hexTable
              | false => false)
            (nibbleChar_mem (lowNibble byte) (lowNibble_lt byte)))
          ih))

theorem bitAt_setBitAt_ne :
    ∀ (bits : List Bool) (index other : Nat),
      ¬(index = other) → bitAt (setBitAt bits index) other = bitAt bits other :=
  List.rec
    (motive := fun bs : List Bool =>
      ∀ index other : Nat, ¬(index = other) → bitAt (setBitAt bs index) other = bitAt bs other)
    (fun _ _ _ => Eq.refl false)
    (fun (b : Bool) (rest : List Bool)
        (ih : ∀ index other : Nat, ¬(index = other) →
          bitAt (setBitAt rest index) other = bitAt rest other) =>
      Nat.rec
        (motive := fun i : Nat =>
          ∀ other : Nat, ¬(i = other) →
            bitAt (setBitAt (b :: rest) i) other = bitAt (b :: rest) other)
        (fun (other : Nat) =>
          Nat.rec
            (motive := fun o : Nat =>
              ¬(0 = o) → bitAt (setBitAt (b :: rest) 0) o = bitAt (b :: rest) o)
            (fun h => absurd (Eq.refl 0) h)
            (fun (o : Nat)
                (_ : ¬(0 = o) → bitAt (setBitAt (b :: rest) 0) o = bitAt (b :: rest) o)
                (_ : ¬(0 = o + 1)) => Eq.refl (bitAt rest o))
            other)
        (fun (i : Nat)
            (_ : ∀ other : Nat, ¬(i = other) →
              bitAt (setBitAt (b :: rest) i) other = bitAt (b :: rest) other)
            (other : Nat) =>
          Nat.rec
            (motive := fun o : Nat =>
              ¬(i + 1 = o) → bitAt (setBitAt (b :: rest) (i + 1)) o = bitAt (b :: rest) o)
            (fun _ => Eq.refl b)
            (fun (o : Nat)
                (_ : ¬(i + 1 = o) →
                  bitAt (setBitAt (b :: rest) (i + 1)) o = bitAt (b :: rest) o)
                (h : ¬(i + 1 = o + 1)) =>
              ih i o (fun heq : i = o => h (congrArg (fun n : Nat => n + 1) heq)))
            other))

theorem bitAt_clearBitAt_ne :
    ∀ (bits : List Bool) (index other : Nat),
      ¬(index = other) → bitAt (clearBitAt bits index) other = bitAt bits other :=
  List.rec
    (motive := fun bs : List Bool =>
      ∀ index other : Nat, ¬(index = other) → bitAt (clearBitAt bs index) other = bitAt bs other)
    (fun _ _ _ => Eq.refl false)
    (fun (b : Bool) (rest : List Bool)
        (ih : ∀ index other : Nat, ¬(index = other) →
          bitAt (clearBitAt rest index) other = bitAt rest other) =>
      Nat.rec
        (motive := fun i : Nat =>
          ∀ other : Nat, ¬(i = other) →
            bitAt (clearBitAt (b :: rest) i) other = bitAt (b :: rest) other)
        (fun (other : Nat) =>
          Nat.rec
            (motive := fun o : Nat =>
              ¬(0 = o) → bitAt (clearBitAt (b :: rest) 0) o = bitAt (b :: rest) o)
            (fun h => absurd (Eq.refl 0) h)
            (fun (o : Nat)
                (_ : ¬(0 = o) → bitAt (clearBitAt (b :: rest) 0) o = bitAt (b :: rest) o)
                (_ : ¬(0 = o + 1)) => Eq.refl (bitAt rest o))
            other)
        (fun (i : Nat)
            (_ : ∀ other : Nat, ¬(i = other) →
              bitAt (clearBitAt (b :: rest) i) other = bitAt (b :: rest) other)
            (other : Nat) =>
          Nat.rec
            (motive := fun o : Nat =>
              ¬(i + 1 = o) → bitAt (clearBitAt (b :: rest) (i + 1)) o = bitAt (b :: rest) o)
            (fun _ => Eq.refl b)
            (fun (o : Nat)
                (_ : ¬(i + 1 = o) →
                  bitAt (clearBitAt (b :: rest) (i + 1)) o = bitAt (b :: rest) o)
                (h : ¬(i + 1 = o + 1)) =>
              ih i o (fun heq : i = o => h (congrArg (fun n : Nat => n + 1) heq)))
            other))

theorem popCountBits_setBitAt :
    ∀ (bits : List Bool) (index : Nat),
      index < listLength bits → bitAt bits index = false →
        popCountBits (setBitAt bits index) = popCountBits bits + 1 :=
  List.rec
    (motive := fun bs : List Bool =>
      ∀ index : Nat,
        index < listLength bs → bitAt bs index = false →
          popCountBits (setBitAt bs index) = popCountBits bs + 1)
    (fun (index : Nat) (h : index < 0) => absurd h (Nat.not_lt_zero index))
    (fun (b : Bool) (rest : List Bool)
        (ih : ∀ index : Nat,
          index < listLength rest → bitAt rest index = false →
            popCountBits (setBitAt rest index) = popCountBits rest + 1) =>
      Nat.rec
        (motive := fun n : Nat =>
          n < listLength rest + 1 → bitAt (b :: rest) n = false →
            popCountBits (setBitAt (b :: rest) n) = popCountBits (b :: rest) + 1)
        (fun (_ : 0 < listLength rest + 1) (hb : b = false) =>
          Eq.symm (congrArg (fun t : Bool => popCountBits (t :: rest) + 1) hb))
        (fun (n : Nat)
            (_ : n < listLength rest + 1 → bitAt (b :: rest) n = false →
              popCountBits (setBitAt (b :: rest) n) = popCountBits (b :: rest) + 1)
            (hn : n + 1 < listLength rest + 1) (hbit : bitAt rest n = false) =>
          Bool.rec
            (motive := fun t : Bool =>
              popCountBits (t :: setBitAt rest n) = popCountBits (t :: rest) + 1)
            (ih n (Nat.lt_of_succ_lt_succ hn) hbit)
            (congrArg (fun m : Nat => m + 1) (ih n (Nat.lt_of_succ_lt_succ hn) hbit))
            b))

theorem natListEq_sound :
    ∀ left right : List Nat, natListEq left right = true → left = right :=
  List.rec
    (motive := fun xs : List Nat => ∀ ys : List Nat, natListEq xs ys = true → xs = ys)
    (fun ys =>
      List.rec
        (motive := fun zs : List Nat => natListEq [] zs = true → [] = zs)
        (fun _ => Eq.refl ([] : List Nat))
        (fun (_ : Nat) (_ : List Nat) _ h => Bool.noConfusion h)
        ys)
    (fun (a : Nat) (as : List Nat)
        (ih : ∀ ys : List Nat, natListEq as ys = true → as = ys) (ys : List Nat) =>
      List.rec
        (motive := fun zs : List Nat => natListEq (a :: as) zs = true → a :: as = zs)
        (fun h => Bool.noConfusion h)
        (fun (b : Nat) (bs : List Nat) _ =>
          Bool.rec
            (motive := fun t : Bool =>
              Nat.beq a b = t →
                (match t with | true => natListEq as bs | false => false) = true →
                  a :: as = b :: bs)
            (fun _ (hbad : false = true) => Bool.noConfusion hbad)
            (fun (hbeq : Nat.beq a b = true) (htail : natListEq as bs = true) =>
              Eq.trans
                (congrArg (fun t : List Nat => a :: t) (ih bs htail))
                (congrArg (fun n : Nat => n :: bs) (Nat.eq_of_beq_eq_true hbeq)))
            (Nat.beq a b)
            (Eq.refl (Nat.beq a b)))
        ys)

theorem lookupAssoc_removeAssoc_ne {β : Type v} :
    ∀ (table : List (List Nat × β)) (key other : List Nat),
      natListEq key other = false →
        lookupAssoc (removeAssoc table key) other = lookupAssoc table other :=
  fun table =>
    List.rec
      (motive := fun t : List (List Nat × β) =>
        ∀ key other : List Nat,
          natListEq key other = false →
            lookupAssoc (removeAssoc t key) other = lookupAssoc t other)
      (fun _ _ _ => Eq.refl none)
      (fun (entry : List Nat × β) (rest : List (List Nat × β))
          (ih : ∀ key other : List Nat,
            natListEq key other = false →
              lookupAssoc (removeAssoc rest key) other = lookupAssoc rest other)
          (key other : List Nat) (hne : natListEq key other = false) =>
        Bool.rec
          (motive := fun t : Bool =>
            natListEq entry.1 key = t →
              lookupAssoc
                  (match t with
                    | true => removeAssoc rest key
                    | false => entry :: removeAssoc rest key)
                  other =
                lookupAssoc (entry :: rest) other)
          (fun _ =>
            Bool.rec
              (motive := fun t : Bool =>
                (match t with
                  | true => some entry.2
                  | false => lookupAssoc (removeAssoc rest key) other) =
                  (match t with
                    | true => some entry.2
                    | false => lookupAssoc rest other))
              (ih key other hne)
              (Eq.refl (some entry.2))
              (natListEq entry.1 other))
          (fun hmatch =>
            Eq.trans (ih key other hne)
              (Eq.symm
                (congrArg
                  (fun t : Bool =>
                    match t with
                    | true => some entry.2
                    | false => lookupAssoc rest other)
                  (Eq.trans
                    (congrArg (fun k : List Nat => natListEq k other)
                      (natListEq_sound entry.1 key hmatch))
                    hne))))
          (natListEq entry.1 key)
          (Eq.refl (natListEq entry.1 key)))
      table

theorem rankedLess_asymm (left right : Nat) (h : rankedLess left right = true) :
    rankedLess right left = false :=
  Bool.rec
    (motive := fun t : Bool => rankedLess right left = t → rankedLess right left = false)
    (fun hfalse => hfalse)
    (fun htrue =>
      absurd (Nat.le_of_ble_eq_true htrue) (Nat.lt_asymm (Nat.le_of_ble_eq_true h)))
    (rankedLess right left)
    (Eq.refl (rankedLess right left))


end Jaide
