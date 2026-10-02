set_option autoImplicit false

universe u

class RSFField (α : Type u) extends Add α, Sub α, Mul α, Div α where
  zero : α
  half : α
  scale : α
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
  scale_sq : scale * scale = half
  double_half : ∀ a : α, (a + a) * half = a
  mul_div_cancel : ∀ a b : α, b ≠ zero → a * b / b = a
  div_mul_cancel : ∀ a b : α, b ≠ zero → a / b * b = a

namespace RSFField

variable {α : Type u} [RSFField α]

theorem sub_add_self (a b : α) : a - b + (a + b) = a + a :=
  calc a - b + (a + b)
      = a + (a + b) - b := RSFField.sub_add a b (a + b)
    _ = a + a + b - b := congrArg (fun t => t - b) (RSFField.add_assoc a a b).symm
    _ = a + a := RSFField.add_sub_cancel (a + a) b

theorem add_sub_self (a b : α) : a + b - (a - b) = b + b :=
  calc a + b - (a - b)
      = a + b - a + b := RSFField.sub_sub_eq_add (a + b) a b
    _ = b + a - a + b := congrArg (fun t => t - a + b) (RSFField.add_comm a b)
    _ = b + b := congrArg (fun t => t + b) (RSFField.add_sub_cancel b a)

theorem add_sub_swap (a b : α) : a + b - (b - a) = a + a :=
  calc a + b - (b - a)
      = a + b - b + a := RSFField.sub_sub_eq_add (a + b) b a
    _ = a + a := congrArg (fun t => t + a) (RSFField.add_sub_cancel a b)

theorem add_add_self (a b : α) : a + b + (b - a) = b + b :=
  calc a + b + (b - a)
      = a + b + b - a := RSFField.add_sub (a + b) b a
    _ = a + (b + b) - a := congrArg (fun t => t - a) (RSFField.add_assoc a b b)
    _ = b + b + a - a := congrArg (fun t => t - a) (RSFField.add_comm a (b + b))
    _ = b + b := RSFField.add_sub_cancel (b + b) a

def unitModel : RSFField Unit where
  add _ _ := ()
  sub _ _ := ()
  mul _ _ := ()
  div _ _ := ()
  zero := ()
  half := ()
  scale := ()
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
  scale_sq := rfl
  double_half _ := rfl
  mul_div_cancel _ _ _ := rfl
  div_mul_cancel _ _ _ := rfl

end RSFField
