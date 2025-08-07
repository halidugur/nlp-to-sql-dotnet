using Microsoft.AspNetCore.Mvc;
using stajprojem.Data;
using stajprojem.Models;

namespace stajprojem.Controllers
{
    public class SuppliersController : Controller
    {
        private readonly AppDbContext _context;

        public SuppliersController(AppDbContext context)
        {
            _context = context;
        }

        public IActionResult Index()
        {
            var suppliers = _context.Suppliers.ToList();
            return View(suppliers);
        }
    }
}
