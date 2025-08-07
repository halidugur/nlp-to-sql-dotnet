using Microsoft.AspNetCore.Mvc;
using stajprojem.Data;
using stajprojem.Models;

namespace stajprojem.Controllers
{
    public class OrdersController : Controller
    {
        private readonly AppDbContext _context;

        public OrdersController(AppDbContext context)
        {
            _context = context;
        }

        public IActionResult Index()
        {
            var orders = _context.Orders.ToList();
            return View(orders);
        }
    }
}
