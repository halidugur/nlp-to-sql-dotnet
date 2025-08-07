using Microsoft.AspNetCore.Mvc;
using stajprojem.Data;
using stajprojem.Models;

namespace stajprojem.Controllers
{
    public class PurchaseOrdersController : Controller
    {
        private readonly AppDbContext _context;

        public PurchaseOrdersController(AppDbContext context)
        {
            _context = context;
        }

        public IActionResult Index()
        {
            var purchaseOrders = _context.PurchaseOrders.ToList();
            return View(purchaseOrders);
        }
    }
}
