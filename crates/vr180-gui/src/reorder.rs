//! A list whose rows are put in another order by dragging a row's handle.
//!
//! Each row draws its [`handle`] and, once it is laid out, hands its
//! rectangle to [`drop_target`]: while a row of the same list is dragged
//! over it a line shows where that row would land, and letting go there
//! returns the [`Move`] to make.

use egui::{Color32, Id, Rect, RichText, Sense, Stroke, Ui};

/// A row taken out of place `from` and put so that it is at place `to` of
/// the list that results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub from: usize,
    pub to: usize,
}

impl Move {
    /// Where the row now at place `index` is after the move.
    pub fn place_of(self, index: usize) -> usize {
        let Move { from, to } = self;
        if index == from {
            to
        } else if from < index && index <= to {
            index - 1
        } else if to <= index && index < from {
            index + 1
        } else {
            index
        }
    }

    /// Make the move in `list`.
    pub fn apply<T>(self, list: &mut Vec<T>) {
        if self.from < list.len() && self.to < list.len() && self.from != self.to {
            let row = list.remove(self.from);
            list.insert(self.to, row);
        }
    }
}

/// What a dragged row carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Dragged {
    list: Id,
    index: usize,
}

/// The mark a row is dragged by (one the built-in fonts hold: a test checks).
pub const HANDLE: &str = "☰";

/// The handle of row `index` of `list`: the row is dragged by it.
pub fn handle(ui: &mut Ui, list: Id, index: usize) -> egui::Response {
    ui.dnd_drag_source(list.with(("row", index)), Dragged { list, index }, |ui| {
        ui.label(RichText::new(HANDLE).color(Color32::GRAY));
    }).response
}

/// Row `index` of `list`, laid out at `row`: the place a dragged row would
/// take is drawn, and the move is returned when one is let go here.
pub fn drop_target(ui: &Ui, list: Id, index: usize, row: Rect) -> Option<Move> {
    let response = ui.interact(row, list.with(("drop", index)), Sense::hover());
    let dragged = *response.dnd_hover_payload::<Dragged>()?;
    if dragged.list != list {
        return None;
    }
    // Above the row's middle the dragged row goes before it, below after.
    let pointer_y = ui.ctx().pointer_interact_pos().map_or(row.center().y, |p| p.y);
    let before = pointer_y < row.center().y;
    let slot = if before { index } else { index + 1 };
    let to = if slot > dragged.index { slot - 1 } else { slot };
    if to != dragged.index {
        let y = if before { row.top() } else { row.bottom() };
        ui.painter().hline(row.x_range(), y, Stroke::new(2.0, Color32::from_rgb(90, 166, 255)));
    }
    response.dnd_release_payload::<Dragged>()?;
    (to != dragged.index).then_some(Move { from: dragged.index, to })
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{pos2, vec2, Event, PointerButton, Pos2, RawInput};

    #[test]
    fn a_move_renumbers_the_rows_between() {
        // Row 1 of five goes to place 3: rows 2 and 3 step up.
        let m = Move { from: 1, to: 3 };
        assert_eq!((0..5).map(|i| m.place_of(i)).collect::<Vec<_>>(), vec![0, 3, 1, 2, 4]);
        let mut list = vec!['a', 'b', 'c', 'd', 'e'];
        m.apply(&mut list);
        assert_eq!(list, vec!['a', 'c', 'd', 'b', 'e']);
        // Back up: row 3 to place 0.
        let m = Move { from: 3, to: 0 };
        assert_eq!((0..5).map(|i| m.place_of(i)).collect::<Vec<_>>(), vec![1, 2, 3, 0, 4]);
        let mut list = vec!['a', 'b', 'c', 'd', 'e'];
        m.apply(&mut list);
        assert_eq!(list, vec!['d', 'a', 'b', 'c', 'e']);
        // Every row's new place is where `apply` puts it.
        for (from, to) in [(0, 4), (4, 0), (2, 2), (1, 2), (2, 1)] {
            let m = Move { from, to };
            let mut list: Vec<usize> = (0..5).collect();
            m.apply(&mut list);
            for (old, row) in (0..5).map(|i| (i, m.place_of(i))) {
                assert_eq!(list[row], old, "{m:?}");
            }
        }
    }

    #[test]
    fn the_handle_is_a_glyph_the_built_in_fonts_hold() {
        let ctx = egui::Context::default();
        let _ = ctx.run_ui(RawInput::default(), |_| {});
        let font = egui::FontId::default();
        assert!(HANDLE.chars().all(|c| ctx.fonts_mut(|f| f.has_glyph(&font, c))), "{HANDLE}");
    }

    /// A list of `n` rows, one frame: where each row and its handle were
    /// laid out, and the move that was made this frame.
    fn frame(ctx: &egui::Context, time: f64, events: Vec<Event>, n: usize) -> (Vec<(Rect, Pos2)>, Option<Move>) {
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 400.0))),
            time: Some(time),
            events,
            ..Default::default()
        };
        let mut rows = Vec::new();
        let mut made = None;
        let _ = ctx.run_ui(input, |ui| {
            let list = Id::new("rows");
            for i in 0..n {
                let mut grip = Pos2::ZERO;
                let row = ui.horizontal(|ui| {
                    grip = handle(ui, list, i).rect.center();
                    ui.label(format!("row {i}"));
                }).response.rect;
                rows.push((row, grip));
                if let Some(m) = drop_target(ui, list, i, row) {
                    made = Some(m);
                }
            }
        });
        (rows, made)
    }

    /// Drag row `from` by its handle and let go at `drop_y` of row `over`
    /// (0 = its top, 1 = its bottom).
    fn drag(n: usize, from: usize, over: usize, drop_y: f32) -> Option<Move> {
        let ctx = egui::Context::default();
        let mut t = 0.0;
        let mut step = |events: Vec<Event>| {
            t += 1.0 / 60.0;
            frame(&ctx, t, events, n)
        };
        let (rows, _) = step(vec![]);
        let grip = rows[from].1;
        let target = rows[over].0;
        let end = pos2(target.center().x, target.top() + target.height() * drop_y);
        let button = |pos, pressed| Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Default::default() };
        step(vec![Event::PointerMoved(grip)]);
        step(vec![button(grip, true)]);
        // Far enough to be a drag, then on to the target in two steps.
        step(vec![Event::PointerMoved(grip + vec2(0.0, 12.0))]);
        step(vec![Event::PointerMoved(pos2(end.x, (grip.y + end.y) * 0.5))]);
        step(vec![Event::PointerMoved(end)]);
        assert!(step(vec![]).1.is_none(), "nothing moves before the row is let go");
        step(vec![button(end, false)]).1
    }

    #[test]
    fn a_row_is_dragged_to_its_new_place() {
        // Onto the lower half of row 3: after it.
        assert_eq!(drag(5, 0, 3, 0.8), Some(Move { from: 0, to: 3 }));
        // Onto the upper half of row 3: before it.
        assert_eq!(drag(5, 0, 3, 0.2), Some(Move { from: 0, to: 2 }));
        // Upwards.
        assert_eq!(drag(5, 4, 1, 0.2), Some(Move { from: 4, to: 1 }));
        assert_eq!(drag(5, 4, 1, 0.8), Some(Move { from: 4, to: 2 }));
        // Let go where it already is: no move.
        assert_eq!(drag(5, 2, 2, 0.5), None);
        assert_eq!(drag(5, 2, 3, 0.2), None);
        assert_eq!(drag(5, 2, 1, 0.8), None);
    }
}
